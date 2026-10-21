"""Make outputs_<new model_name>/ for re-evaluating an existing flow with another MAP setting: config with the new
model_name / MAP settings / test files, and symlinks to the trained model and its normalisation."""
import os, re, sys
src, dest_parent, new_name, map_method, chunk, tests, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6].split(','), sys.argv[7].split(',')
txt = open(os.path.join(src, 'config.yaml')).read()
old_name = re.search(r'^  model_name: (\S+)', txt, re.M).group(1)
txt = re.sub(r'^  model_name: \S+', f'  model_name: {new_name}', txt, count=1, flags=re.M)
txt = re.sub(r'^  map_method: \S+', f'  map_method: {map_method}\n  map_n_steps: 200\n  map_n_random_starts: 10\n  chunk_size: {chunk}', txt, count=1, flags=re.M)
txt = re.sub(r'^  test_dataset: .*$', '  test_dataset:  [' + ', '.join(f"'{t}'" for t in tests) + ']', txt, count=1, flags=re.M)
txt = re.sub(r'^  test_output_name: .*$', '  test_output_name: [' + ', '.join(f"'{n}'" for n in names) + ']', txt, count=1, flags=re.M)
out = os.path.join(dest_parent, f'outputs_{new_name}')
os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
open(os.path.join(out, 'config.yaml'), 'w').write(f'# re-evaluation of {os.path.abspath(src)} ({old_name}) with map_method {map_method}\n' + txt)
for f in ('normalization_params.npz', 'plots/best_model.pth'):
    if not os.path.lexists(os.path.join(out, f)):
        os.symlink(os.path.abspath(os.path.join(src, f)), os.path.join(out, f))
print(out)
