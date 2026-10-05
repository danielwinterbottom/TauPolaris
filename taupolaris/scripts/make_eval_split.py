"""Write the evaluation test file for a training config: the WHOLE of an evaluation dataset
(default ppToHToTauTau_AllDM_UnCorr_Jul20, never used in training) after that config's
channel selection (Data.leptonic_mode, plus the tau1/tau2 relabelling for semileptonic),
under the name the training split uses -- i.e. exactly the file the config's
Data.test_dataset points to. This is what full_dataframe_testing: True produced for the
hadronic Jul20 test file; use it for the semileptonic (leptonic_mode 1) and combined (-1)
models.

Memory-bounded: the prepared dataframe is streamed in blocks through the same
apply_channel_selection the training split uses (the selection is event-by-event), so this
runs on the login node -- loading the whole 6.5M-event Jul20 dataframe at once would need
~25-40 GB.

Equivalence with the training split's own (whole-file) test file, checked on a mini sample:
identical for leptonic_mode 0 and -1 (rows, order, columns, index); for leptonic_mode 1 the
same events and columns but in a different ORDER, because convert_semileptonic_df shuffles
with a fixed seed and here it sees one block at a time. Every evaluation output is a
per-event aggregate, so this changes nothing in the results; only row-by-row alignment with a
file made the other way would differ.

    python taupolaris/scripts/make_eval_split.py -c <training config> [--dataset ppToHToTauTau_AllDM_UnCorr_Jul20]
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, os.getcwd())
from taupolaris.python.DataProcessing import apply_channel_selection


def split_name(data):
    """Same naming as get_train_val_test_datasets."""
    extra = 'cartesian' if data['coordinates'] == 'standard' else str(data['coordinates'])
    if data.get('leptonic_mode', -1) >= 0:
        extra += f"_leptonic_mode_{data['leptonic_mode']}"
    if data.get('inc_three_prongs', False):
        extra += '_inc_three_prongs'
    if data['use_transformer']:
        extra += '_transformer'
    return f'test_dataframe_{extra}.parquet'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c', required=True)
    ap.add_argument('--dataset', default='ppToHToTauTau_AllDM_UnCorr_Jul20')
    ap.add_argument('--batch_size', type=int, default=250_000)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.dataset in cfg['Data'].get('datasets', []):
        raise SystemExit(f'{args.dataset} is one of this config\'s TRAINING datasets -- its test split comes '
                         'from the training job; this script is for evaluation-only datasets (default Jul20).')
    data = copy.deepcopy(cfg['Data'])
    data['use_transformer'] = cfg['SetupNN'].get('use_transformer', False)
    if data['coordinates'] not in ('onorm', 'standard'):
        raise SystemExit(f"coordinates={data['coordinates']} not supported here (onorm/standard only)")

    d = os.path.join(data['output_dir'], args.dataset)
    src = os.path.join(d, 'full_onorm_dataframe.parquet')
    if not os.path.exists(src):
        src = os.path.join(d, 'full_onorm_angular_dataframe.parquet')
    out = os.path.join(d, split_name(data))
    expected = [t for t in cfg['Data']['test_dataset'] if args.dataset in t]
    if expected and os.path.normpath(expected[0]) != os.path.normpath(out):
        print(f">> WARNING: the config's test_dataset is {expected[0]}, but this writes {out}")
    print(f'>> {src} -> {out} (leptonic_mode {data.get("leptonic_mode", -1)}), streaming in blocks of {args.batch_size}')

    tmp = f'{out}.tmp.{os.getpid()}'
    pf = pq.ParquetFile(src)
    writer, schema, n_in, n_out, offset, t0 = None, None, 0, 0, 0, time.time()
    try:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            df = batch.to_pandas()
            df.index = np.arange(offset, offset + len(df))   # row position in the full dataframe, as a whole-file read would give
            offset += len(df); n_in += len(df)
            df['dataset'] = args.dataset
            df = apply_channel_selection(df, data)
            if len(df) == 0:
                continue
            if data.get('leptonic_mode', -1) == 1:
                # the semileptonic relabelling returns a fresh 0..n-1 index (after its fixed-seed
                # shuffle); continue it across blocks so the file is indexed 0..N-1 like a whole-file split
                df.index = np.arange(n_out, n_out + len(df))
            table = pa.Table.from_pandas(df, preserve_index=True)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema)
            else:
                table = table.cast(schema)
            writer.write_table(table)
            n_out += len(df)
            print(f'   {n_in} read, {n_out} kept ({time.time() - t0:.0f}s)', flush=True)
        if writer is None:
            raise SystemExit('no events passed the selection')
        writer.close()
        os.replace(tmp, out)
    except BaseException:
        if writer is not None:
            writer.close()
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f'>> wrote {out}: {n_out} of {n_in} events')


if __name__ == '__main__':
    main()
