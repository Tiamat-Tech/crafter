import json
import pathlib
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def binning(xs, ys, borders, reducer=np.nanmean, fill='nan'):
  xs, ys = np.array(xs), np.array(ys)
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  binned = []
  with warnings.catch_warnings():  # Empty borders become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    for start, stop in zip(borders[:-1], borders[1:]):
      left = (xs <= start).sum()
      right = (xs <= stop).sum()
      if left < right:
        value = reducer(ys[left:right])
      elif binned:
        value = {'nan': np.nan, 'last': binned[-1]}[fill]
      else:
        value = np.nan
      binned.append(value)
  return borders[1:], np.array(binned)


def plot_reward(indir, legend, colors, cols=4, budget=5e6):
  print('Loading runs:')
  runs = []
  for filename in pathlib.Path(indir).glob('**/*.json'):
    loaded = json.loads(filename.read_text())
    for run in [loaded] if isinstance(loaded, dict) else loaded:
      print(f'- {run["method"]} seed {run["seed"]}', flush=True)
      if run['xs'][-1] < 0.99 * budget:
        print(f'  Contains only {run["xs"][-1]} steps!')
      runs.append(run)
  print('')
  borders = np.arange(0, budget, 1e5)

  fig, ax = plt.subplots(figsize=(3.2, 2.8))
  ax.set_title('Episode Reward')
  for j, (method, label) in enumerate(legend.items()):
    relevant = [run for run in runs if run['method'] == method]
    xs = np.concatenate([run['xs'] for run in relevant])
    ys = np.concatenate([run['score'] for run in relevant])
    means = binning(xs, ys, borders, np.nanmean)[1]
    stds = binning(xs, ys, borders, np.nanstd)[1]
    kwargs = dict(alpha=0.2, linewidths=0, color=colors[j], zorder=10 - j)
    ax.fill_between(borders[1:], means - stds, means + stds, **kwargs)
    ax.plot(borders[1:], means, label=label, color=colors[j], zorder=100 - j)
  ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  ax.xaxis.set_major_locator(ticker.MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
  ax.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
  ax.legend(loc='upper left', frameon=False, handlelength=1)
  fig.tight_layout()
  return fig


filename = pathlib.Path('results/score.pdf')
filename.parent.mkdir(exist_ok=True, parents=True)
legend = {'dreamerv2': 'DreamerV2', 'ppo': 'PPO', 'random': 'Random'}
colors = ['#377eb8', '#4daf4a', '#a65628', '#984ea3', '#e41a1c']
plot_reward('runs', legend, colors).savefig(filename)
print(f'Saved {filename}')
