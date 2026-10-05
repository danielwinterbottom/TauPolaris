import argparse
import numpy as np
import yaml


def compute_total_asymmetry(asymmetry_dict):
    n = np.array([entry['N'] for entry in asymmetry_dict.values()], dtype=float)
    asymmetry = np.array([entry['asymmetry'] for entry in asymmetry_dict.values()], dtype=float)
    return np.sqrt(np.sum(n * asymmetry**2) / np.sum(n))


def load_asymmetry_yaml(path):
    """Plain-float files (make_asymmetry_yaml.py) load with safe_load. plot_phiCP.py dumps
    numpy float32 scalars through !!python/object/apply tags, which need unsafe_load and,
    worse, name the numpy module of the writer (numpy._core.* for numpy >= 2), so a file
    written under one numpy version does not load under another. Fall back to decoding the
    little-endian float32 payload directly in that case."""
    with open(path, 'r') as f:
        text = f.read()
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        pass
    try:
        return yaml.unsafe_load(text)
    except Exception:
        import base64, re
        import numpy as np
        out = {}
        for m in re.finditer(r"^(DM\d+DM\d+):\n  N: (\d+)\n  asymmetry:.*?!!binary \|\n    (\S+)", text, re.S | re.M):
            out[m.group(1)] = {'N': int(m.group(2)),
                               'asymmetry': float(np.frombuffer(base64.b64decode(m.group(3)), dtype='<f4')[0])}
        if not out:
            raise
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_files', nargs='+', help="Paths to asymmetry yaml files (e.g. "
                                           "asymmetry_had_Polaris.yaml asymmetry_lep_Polaris.yaml) "
                                           "produced by plot_phiCP.py, each keyed by DM combination "
                                           "with 'N' and 'asymmetry' entries. Combined into one total.")
    args = parser.parse_args()

    asymmetry_dict = {}
    for path in args.yaml_files:
        channel_dict = load_asymmetry_yaml(path)
        overlap = asymmetry_dict.keys() & channel_dict.keys()
        if overlap:
            raise ValueError(f"Duplicate DM-combination keys found across input files: {sorted(overlap)}")
        print(f"-- {path} --")
        for key, entry in channel_dict.items():
            print(f"{key}: N={entry['N']}, asymmetry={float(entry['asymmetry']):.4f}")
        asymmetry_dict.update(channel_dict)

    total_asymmetry = compute_total_asymmetry(asymmetry_dict)
    print(f"Total asymmetry: {total_asymmetry:.4f}")


if __name__ == '__main__':
    main()
