import argparse
import logging
import math
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from add_fastmtt import build_visible_leg, build_leptonic_leg, get_leptonic_decay_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# patched ClassicSVfit + pybind wrapper, built from the TIDAL fork (see SOURCE_COMMIT.txt there)
# against the DiTauEntanglement env's ROOT. The patch fixes MeasuredTauLepton::operator=, which
# did not copy the visible mass, so unpatched builds fit with uninitialised visible masses.
# The wrapper's RPATH points at this ROOT: loading any other ROOT in the same process mixes two
# ROOT versions, which also makes the fits silently irreproducible.
svfit_dir = "/vols/cms/lcr119/offline/HiggsCP/ClassicSVfit_patched"
svfit_root_lib_dir = "/vols/cms/lcr119/miniforge3/envs/DiTauEntanglement/lib"

# MET covariance is assumed constant across events (same as add_fastmtt.py).
met_cov_xx = 184.0
met_cov_xy = 0.0
met_cov_yy = 184.0

# di-tau mass constraint: only the constrained fit is run.
mass_constraint = 125.0

# exponent of the 1/m^kappa regularisation term, values from TIDAL run_svfit.py.
kappa = {'tt': 5.0, 'lt': 4.0}

# events per condor job (~0.1 s per event).
chunk_size = 10_000

# ClassicSVfit MeasuredTauLepton::kDecayType
svfit_type_had, svfit_type_ele, svfit_type_mu = 1, 2, 3

m_ele, m_muon = 0.51100e-3, 0.10566

suffix = '_constraint'


def get_args():
    parser = argparse.ArgumentParser(description="Add mass-constrained ClassicSVfit tau 4-vectors to a ditau dataframe (parquet). "
                                                 "Workflow: --submit (condor chunk jobs) -> --merge.")
    parser.add_argument('--input_file', type=str, required=True, help="Input parquet file.")
    parser.add_argument('--output_file', type=str, required=True, help="Final merged output parquet file (must not exist, must differ from input_file).")
    parser.add_argument('--channel', type=str, required=True, choices=['tt', 'lt'], help="tt: both legs hadronic (taup/taun columns). lt: tau1 leptonic, tau2 hadronic (tau1/tau2 columns).")
    parser.add_argument('--chunk_size', type=int, default=chunk_size, help="Events per condor job (must match between --submit and --merge).")
    parser.add_argument('--runtime', type=int, default=3 * 3600, help="Condor MaxRuntime in seconds.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--submit', action='store_true', help="Write and submit one condor job per chunk.")
    mode.add_argument('--merge', action='store_true', help="Merge finished chunks into output_file.")
    mode.add_argument('--chunk_index', type=int, help="Process a single chunk (used by the condor jobs).")
    return parser.parse_args()


def get_chunk_dir(output_file):
    return os.path.splitext(output_file)[0] + '_chunks'


def get_chunk_path(output_file, index):
    return os.path.join(get_chunk_dir(output_file), f'chunk{index}.parquet')


def refuse_if_exists(path):
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists, refusing to overwrite. Delete it by hand to rerun.")


def get_input_columns(channel):
    # the columns read by the leg builders in add_fastmtt.py, get_hadronic_decay_mode and the MET.
    visible = [f'reco_{{leg}}_{p}_{c}' for p in ('charged', 'pizero1') for c in ('px', 'py', 'pz', 'e')]
    visible += ['reco_{leg}_is3prong', 'reco_{leg}_npizero']
    if channel == 'tt':
        columns = [c.format(leg=leg) for leg in ('taup', 'taun') for c in visible]
    else:  # lt
        columns = [f'reco_tau1_lep_{c}' for c in ('px', 'py', 'pz', 'e')] + ['reco_tau1_ismuon']
        columns += [c.format(leg='tau2') for c in visible]
    return columns + ['reco_met_px', 'reco_met_py']


def read_rows(input_file, first, n, columns=None):
    # read only the row groups overlapping [first, first + n), then slice to the exact range.
    parquet_file = pq.ParquetFile(input_file)
    groups, group_start, offset = [], 0, None
    for i in range(parquet_file.num_row_groups):
        group_rows = parquet_file.metadata.row_group(i).num_rows
        if group_start + group_rows > first and group_start < first + n:
            if offset is None:
                offset = first - group_start
            groups.append(i)
        group_start += group_rows
    return parquet_file.read_row_groups(groups, columns=columns).slice(offset, n)


def get_n_chunks(input_file, size):
    nevents = pq.ParquetFile(input_file).metadata.num_rows
    return nevents, math.ceil(nevents / size)


def import_svfit():
    sys.path.insert(0, svfit_dir)
    import ROOT
    root_core = os.path.realpath(ROOT.gSystem.DynamicPathName('libCore'))
    if os.path.dirname(root_core) != os.path.realpath(svfit_root_lib_dir):
        raise RuntimeError(f"Loaded ROOT from {root_core}, but ClassicSVfit is built against {svfit_root_lib_dir}: "
                           "run in the DiTauEntanglement env (mamba activate DiTauEntanglement).")
    from TauAnalysis.ClassicSVfit.wrapper.pybind_wrapper import ClassicSVfit, TauTauHistogramAdapter, MeasuredTauLepton
    return ROOT, ClassicSVfit, TauTauHistogramAdapter, MeasuredTauLepton


def get_hadronic_decay_mode(df, leg):
    # no decay mode column in the dataframe: rebuild the HPS-like DM (0, 1, 10, 11).
    # ClassicSVfit only uses it to fix the visible mass to m_pi for DM0.
    is3prong = df[f'reco_{leg}_is3prong'].to_numpy().astype(bool)
    haspizero = df[f'reco_{leg}_npizero'].to_numpy() > 0
    return (10 * is3prong + haspizero).astype(int)


def delta_r(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi
    return math.hypot(eta1 - eta2, dphi)


def add_svfit(df, channel):
    ROOT, ClassicSVfit, TauTauHistogramAdapter, MeasuredTauLepton = import_svfit()
    nevents = len(df)

    # step 1: visible legs, built exactly as in add_fastmtt.py.
    # step 2: ClassicSVfit lepton type + decay mode per leg.
    if channel == 'tt':
        pt_1, eta_1, phi_1, mass_1 = build_visible_leg(df, 'taup')
        pt_2, eta_2, phi_2, mass_2 = build_visible_leg(df, 'taun')
        type_1 = np.full(nevents, svfit_type_had)
        dm_1 = get_hadronic_decay_mode(df, 'taup')
    else:  # lt
        pt_1, eta_1, phi_1, _ = build_leptonic_leg(df, 'tau1')
        is_muon = get_leptonic_decay_type(df, 'tau1') == 1
        type_1 = np.where(is_muon, svfit_type_mu, svfit_type_ele)
        mass_1 = np.where(is_muon, m_muon, m_ele)
        dm_1 = np.full(nevents, -1)
        pt_2, eta_2, phi_2, mass_2 = build_visible_leg(df, 'tau2')
    type_2 = np.full(nevents, svfit_type_had)
    dm_2 = get_hadronic_decay_mode(df, 'taun' if channel == 'tt' else 'tau2')

    # step 3: MET and its (constant) covariance.
    met_x = df['reco_met_px'].to_numpy()
    met_y = df['reco_met_py'].to_numpy()
    cov = ROOT.TMatrixD(2, 2)
    cov[0][0], cov[0][1] = met_cov_xx, met_cov_xy
    cov[1][0], cov[1][1] = met_cov_xy, met_cov_yy

    out_names = [f'SVfit_tau{i}_{c}{suffix}' for i in (1, 2) for c in ('px', 'py', 'pz', 'e')]
    out_names += [f'SVfit_pt_1{suffix}', f'SVfit_pt_2{suffix}', f'SVfit_mass{suffix}', f'SVfit_mass_err{suffix}', f'SVfit_pt{suffix}']
    out = {name: np.full(nevents, -1.0) for name in out_names}
    valid = np.zeros(nevents, dtype=bool)

    # step 4: run the constrained fit event by event.
    for i in tqdm(range(nevents), desc="SVfit", mininterval=30):
        inputs = (pt_1[i], eta_1[i], phi_1[i], mass_1[i], pt_2[i], eta_2[i], phi_2[i], mass_2[i], met_x[i], met_y[i])
        if not np.all(np.isfinite(inputs)) or pt_1[i] <= 0 or pt_2[i] <= 0:
            continue
        try:
            leg1 = MeasuredTauLepton(int(type_1[i]), float(pt_1[i]), float(eta_1[i]), float(phi_1[i]), float(max(mass_1[i], 0.0)), int(dm_1[i]))
            leg2 = MeasuredTauLepton(int(type_2[i]), float(pt_2[i]), float(eta_2[i]), float(phi_2[i]), float(max(mass_2[i], 0.0)), int(dm_2[i]))

            svfit = ClassicSVfit(0)
            svfit.setHistogramAdapter(TauTauHistogramAdapter())
            svfit.addLogM_fixed(True, kappa[channel])
            svfit.setDiTauMassConstraint(mass_constraint)
            svfit.integrate([leg1, leg2], float(met_x[i]), float(met_y[i]), cov)
            if not svfit.isValidSolution():
                continue

            adapter = svfit.getHistogramAdapter()
            fit_a, fit_b = adapter.GetFittedTau1LV(), adapter.GetFittedTau2LV()
            mass, mass_err = adapter.getMass(), adapter.getMassErr()
        except Exception as exc:
            logger.warning(f"Event {i} failed: {exc}")
            continue

        # step 5: ClassicSVfit sorts its legs internally (lepton first, else by descending pt),
        # so match the fitted taus back to our legs by smallest summed delta R.
        dr_keep = delta_r(fit_a.Eta(), fit_a.Phi(), eta_1[i], phi_1[i]) + delta_r(fit_b.Eta(), fit_b.Phi(), eta_2[i], phi_2[i])
        dr_swap = delta_r(fit_b.Eta(), fit_b.Phi(), eta_1[i], phi_1[i]) + delta_r(fit_a.Eta(), fit_a.Phi(), eta_2[i], phi_2[i])
        fit_1, fit_2 = (fit_a, fit_b) if dr_keep <= dr_swap else (fit_b, fit_a)

        # step 6: store the full fitted tau 4-vectors and derived quantities.
        for n, fit in ((1, fit_1), (2, fit_2)):
            out[f'SVfit_tau{n}_px{suffix}'][i] = fit.Px()
            out[f'SVfit_tau{n}_py{suffix}'][i] = fit.Py()
            out[f'SVfit_tau{n}_pz{suffix}'][i] = fit.Pz()
            out[f'SVfit_tau{n}_e{suffix}'][i] = fit.E()
            out[f'SVfit_pt_{n}{suffix}'][i] = fit.Pt()
        out[f'SVfit_mass{suffix}'][i] = mass
        out[f'SVfit_mass_err{suffix}'][i] = mass_err
        out[f'SVfit_pt{suffix}'][i] = (fit_1 + fit_2).Pt()
        valid[i] = True

    logger.info(f"Valid SVfit solutions: {valid.sum()}/{nevents}")
    for name, values in out.items():
        df[name] = values
    df[f'SVfit_valid{suffix}'] = valid
    return df


def run_chunk(args):
    nevents, n_chunks = get_n_chunks(args.input_file, args.chunk_size)
    if not 0 <= args.chunk_index < n_chunks:
        raise ValueError(f"chunk_index {args.chunk_index} out of range [0, {n_chunks})")
    chunk_path = get_chunk_path(args.output_file, args.chunk_index)
    refuse_if_exists(chunk_path)
    os.makedirs(os.path.dirname(chunk_path), exist_ok=True)

    first = args.chunk_index * args.chunk_size
    df = read_rows(args.input_file, first, args.chunk_size, columns=get_input_columns(args.channel)).to_pandas()
    logger.info(f"Chunk {args.chunk_index}: events {first} to {first + len(df)}")
    df = add_svfit(df, args.channel)

    # the chunk keeps only the SVfit columns, plus the global row index and a copy of reco_met_px
    # so that --merge can check the chunk lines up with the input rows.
    df['row_index'] = np.arange(first, first + len(df), dtype=np.int64)
    df = df[['row_index', 'reco_met_px'] + [c for c in df if c.startswith('SVfit_')]]

    # write to a temporary file and rename, so a crashed job never leaves a finished-looking chunk.
    tmp_path = chunk_path + '.tmp'
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp_path)
    refuse_if_exists(chunk_path)
    os.rename(tmp_path, chunk_path)
    logger.info(f"Wrote {chunk_path}")


def submit(args):
    # jobs reuse this python interpreter, so submit from the DiTauEntanglement env.
    refuse_if_exists(args.output_file)
    chunk_dir = get_chunk_dir(args.output_file)
    refuse_if_exists(chunk_dir)
    condor_dir = os.path.join(chunk_dir, 'condor')
    os.makedirs(condor_dir)

    nevents, n_chunks = get_n_chunks(args.input_file, args.chunk_size)
    script = os.path.abspath(__file__)
    for index in range(n_chunks):
        name = f'svfit_chunk{index}'
        sh_path = os.path.join(condor_dir, f'{name}.sh')
        sub_path = os.path.join(condor_dir, f'{name}.sub')
        with open(sh_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"{sys.executable} -u {script} --input_file {os.path.abspath(args.input_file)} "
                    f"--output_file {os.path.abspath(args.output_file)} --channel {args.channel} "
                    f"--chunk_size {args.chunk_size} --chunk_index {index}\n")
        os.chmod(sh_path, 0o755)
        lines = [
            f"executable = {sh_path}",
            "getenv = True",
            f"output = {condor_dir}/{name}_$(CLUSTER).out",
            f"error = {condor_dir}/{name}_$(CLUSTER).err",
            f"log = {condor_dir}/{name}_$(CLUSTER).log",
            f"+MaxRuntime = {args.runtime}",
            "queue",
        ]
        with open(sub_path, 'w') as f:
            f.write("\n".join(lines))
        os.system(f"condor_submit {sub_path}")
    logger.info(f"Submitted {n_chunks} jobs for {nevents} events, chunks will be written to {chunk_dir}")


def merge(args):
    refuse_if_exists(args.output_file)
    nevents, n_chunks = get_n_chunks(args.input_file, args.chunk_size)
    chunk_paths = [get_chunk_path(args.output_file, i) for i in range(n_chunks)]
    missing = [p for p in chunk_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{n_chunks} chunks missing:\n" + "\n".join(missing))

    # stream the input one row group at a time and append the SVfit columns from the chunks, so the
    # output holds every original column unchanged. Written to a temporary file that is only renamed
    # once every row has passed the alignment checks.
    input_file = pq.ParquetFile(args.input_file)
    chunk_tables = (pq.read_table(path) for path in chunk_paths)
    buffer = next(chunk_tables)
    svfit_columns = [c for c in buffer.column_names if c.startswith('SVfit_')]
    schema = input_file.schema_arrow
    for name in svfit_columns:
        schema = schema.append(buffer.schema.field(name))

    tmp_path = args.output_file + '.tmp'
    refuse_if_exists(tmp_path)
    n_written = 0
    try:
        with pq.ParquetWriter(tmp_path, schema) as writer:
            for group in tqdm(range(input_file.num_row_groups), desc="Merging"):
                table = input_file.read_row_group(group)
                n = table.num_rows
                while buffer.num_rows < n:
                    buffer = pa.concat_tables([buffer, next(chunk_tables)])
                svfit, buffer = buffer.slice(0, n), buffer.slice(n)

                expected_index = np.arange(n_written, n_written + n, dtype=np.int64)
                if not np.array_equal(svfit.column('row_index').to_numpy(), expected_index):
                    raise RuntimeError(f"Chunk rows do not line up with input rows {n_written}-{n_written + n}")
                if not np.array_equal(svfit.column('reco_met_px').to_numpy(), table.column('reco_met_px').to_numpy()):
                    raise RuntimeError(f"reco_met_px differs between chunks and input in rows {n_written}-{n_written + n}")

                for name in svfit_columns:
                    table = table.append_column(schema.field(name), svfit.column(name))
                writer.write_table(table)
                n_written += n
        if n_written != nevents or buffer.num_rows != 0 or next(chunk_tables, None) is not None:
            raise RuntimeError(f"Merged {n_written} rows but input has {nevents}, or chunks have extra rows")
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    refuse_if_exists(args.output_file)
    os.rename(tmp_path, args.output_file)
    logger.info(f"Merged {n_chunks} chunks ({n_written} events) into {args.output_file}")


def main():
    args = get_args()
    if os.path.realpath(args.output_file) == os.path.realpath(args.input_file):
        raise ValueError("output_file must differ from input_file!")

    if args.submit:
        submit(args)
    elif args.merge:
        merge(args)
    else:
        run_chunk(args)
        # skip interpreter teardown: destroying the ClassicSVfit/ROOT objects at exit dumps core.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
