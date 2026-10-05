"""Add the frame-aligned input columns (Data.frame_aligned_inputs, see
DataProcessing._add_frame_aligned_features) to ALREADY PREPARED parquet files,
writing augmented copies into a new prepared-data directory.

Why not just re-prepare from ROOT: the train/val/test split is deterministic
(the Sep 3 and Sep 18 split files of prepared_LHC_data_sep01 are byte-identical),
and the columns being added are exact functions of columns already in the files.
Augmenting the existing split files therefore reproduces the same rows in the
same order -- which also keeps the test file row-aligned with the nu-method's
evaluation output -- at a fraction of the cost of a full re-preparation, and
the jobs load them with train.py --loadDS.

Rows are streamed in batches through a ParquetWriter, so memory stays at a few
GB per file whatever its size. Output is written to a temporary name and
renamed at the end, so a partially written file is never mistaken for a
finished one.

Usage
-----
    python taupolaris/scripts/augment_frame_aligned_inputs.py \
        --src prepared_LHC_data_sep01 --dst prepared_LHC_data_sep22 --level 3 \
        --datasets ppToHToTauTau_AllDM_UnCorr_Jul20_ext1_large_even ... \
        --splits train val test --split_suffix onorm_leptonic_mode_0_transformer

    # a single file, keeping its name (e.g. the explicitly named evaluation test file)
    python taupolaris/scripts/augment_frame_aligned_inputs.py --level 3 \
        --file prepared_LHC_data_sep01/ppToHToTauTau_AllDM_UnCorr_Jul20/test_dataframe__leptonic_mode_0_transformer.parquet \
        --dst prepared_LHC_data_sep22
"""
import argparse
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from taupolaris.python.DataProcessing import _add_frame_aligned_features, frame_aligned_feature_names


def augment_file(src, dst, level, prefix='reco_', rows_per_group=500_000, cols_per_pass=40):
    """Two-stage, memory-bounded augmentation.

    The prepared split files are single-row-group parquet files (5.9M-14M rows x
    314 columns). Arrow cannot stream rows out of one row group without decoding
    every requested column chunk in full, so a naive row-batch loop peaks at
    ~3x the file size (16 GB for a 6.4 GB file, 40+ GB for the DM0and1 ones) --
    over the process ceiling on the interactive nodes. Column chunks, however,
    are independent, so the file is instead read a few dozen columns at a time:

      stage 1  read the columns the features need, compute the new columns for
               all rows (kept as float32, a few GB at most)
      stage 2  for each pass of `cols_per_pass` source columns: read them for all
               rows and rewrite them as an intermediate file with row groups of
               `rows_per_group` rows
      stage 3  for each row group i: read group i from every intermediate, append
               the matching slice of the new columns, write it as row group i of
               the output

    Every stage holds at most (cols_per_pass x N rows) plus one row group's worth
    of everything, i.e. a few GB. The result has the same rows in the same order
    and the same columns plus the new ones; only the row-group layout differs,
    which nothing downstream depends on. Output is written to a temporary name
    and renamed at the end.
    """
    import numpy as np
    import resource
    rss = lambda: f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB peak RSS'

    pf = pq.ParquetFile(src)
    # the split files store their (meaningless, post-reset) pandas index as a
    # column; pandas drops it on read and the row-batch version of this script
    # did too, so leave it out here as well
    names = [c for c in pf.schema_arrow.names if c not in ('index', '__index_level_0__')]
    n_rows = pf.metadata.num_rows
    charged_name = 'charged' if f'{prefix}taup_charged_e' in names else 'pi1'
    expected = frame_aligned_feature_names(level, 'taup', prefix) + frame_aligned_feature_names(level, 'taun', prefix)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    tmp = dst + '.tmp'
    t0 = time.time()

    # ---- stage 1: the new columns for all rows, from the few source columns they need
    need = set()
    for tau in ('taup', 'taun'):
        base = f'{prefix}{tau}_'
        need |= {f'{base}{charged_name}_{c}' for c in ('px', 'py', 'pz', 'e')}
        need |= {f'{base}pizero1_{c}' for c in ('px', 'py', 'pz', 'e')}
        need |= {f'{base}sv_{c}' for c in ('x', 'y', 'z')}
        need |= {f'{base}{part}_{c}' for part in ('pi1', 'pi2', 'pi3') for c in ('px', 'py', 'pz')}
        need |= {f'{base}charged_ip{c}' for c in ('x', 'y', 'z')}
    need = sorted(need & set(names))
    src_tab = pq.read_table(src, columns=need)
    new_cols = {c: np.empty(n_rows, dtype=np.float32) for c in expected}
    for start in range(0, n_rows, rows_per_group):
        df = src_tab.slice(start, rows_per_group).to_pandas()
        df = _add_frame_aligned_features(df, prefix, charged_name, level=level)
        for c in expected:
            new_cols[c][start:start + len(df)] = df[c].to_numpy(dtype=np.float32)
    del src_tab, df
    print(f'>>   {os.path.basename(dst)}: new columns computed ({time.time() - t0:.0f}s, {rss()})', flush=True)

    # ---- stage 2: re-chunk the source, a few dozen columns per pass
    parts = []
    for j, start in enumerate(range(0, len(names), cols_per_pass)):
        cols = names[start:start + cols_per_pass]
        part = f'{tmp}.part{j}'
        tab = pq.read_table(src, columns=cols)
        w = pq.ParquetWriter(part, tab.schema, compression='snappy')
        for rs in range(0, n_rows, rows_per_group):
            w.write_table(tab.slice(rs, rows_per_group))
        w.close()
        del tab
        parts.append(part)
    print(f'>>   {os.path.basename(dst)}: {len(parts)} column passes re-chunked ({time.time() - t0:.0f}s, {rss()})', flush=True)

    # ---- stage 3: merge row group by row group
    readers = [pq.ParquetFile(p) for p in parts]
    n_groups = readers[0].metadata.num_row_groups
    assert all(r.metadata.num_row_groups == n_groups for r in readers)
    writer = None
    for g in range(n_groups):
        pieces = [readers[0].read_row_group(g)] + [r.read_row_group(g) for r in readers[1:]]
        arrays, fields = [], []
        for piece in pieces:
            for name, col in zip(piece.column_names, piece.columns):
                arrays.append(col); fields.append(piece.schema.field(name))
        rs = g * rows_per_group
        for c in expected:
            arrays.append(pa.array(new_cols[c][rs:rs + pieces[0].num_rows])); fields.append(pa.field(c, pa.float32()))
        out = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
        if writer is None:
            writer = pq.ParquetWriter(tmp, out.schema, compression='snappy')
        writer.write_table(out)
    writer.close()
    for p in parts:
        os.remove(p)
    if pq.ParquetFile(tmp).metadata.num_rows != n_rows:
        raise RuntimeError(f'{tmp}: row count changed')
    os.replace(tmp, dst)
    print(f'>> {dst}: {n_rows} rows, +{len(expected)} columns, {time.time() - t0:.0f}s, {rss()}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='prepared_LHC_data_sep01')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--level', type=int, required=True, choices=[1, 2, 3])
    ap.add_argument('--datasets', nargs='*', default=[])
    ap.add_argument('--splits', nargs='*', default=['train', 'val', 'test'])
    ap.add_argument('--split_suffix', default='onorm_leptonic_mode_0_transformer')
    ap.add_argument('--file', nargs='*', default=[], help='explicit source files; written under --dst/<dataset dir>/<same name>')
    args = ap.parse_args()

    jobs = []
    for k in args.datasets:
        for split in args.splits:
            name = f'{split}_dataframe_{args.split_suffix}.parquet'
            jobs.append((os.path.join(args.src, k, name), os.path.join(args.dst, k, name)))
    for f in args.file:
        k = os.path.basename(os.path.dirname(os.path.abspath(f)))
        jobs.append((f, os.path.join(args.dst, k, os.path.basename(f))))
    for src, dst in jobs:
        if os.path.exists(dst):
            print(f'>> {dst} exists, skipping', flush=True)
            continue
        augment_file(src, dst, args.level)


if __name__ == '__main__':
    main()
