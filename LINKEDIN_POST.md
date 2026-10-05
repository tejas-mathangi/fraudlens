# LinkedIn post drafts

Two versions. Pick one, don't post both. Lead with
`docs/screenshots/explorer.png` — the clustered graph reads instantly in a feed, where the
overview screenshot turns into unreadable grey mush at thumbnail size.

Replace `<repo-link>` with `https://github.com/tejas-mathangi/fraudlens`.

---

## Version A — the technical one

> I built a fraud detection system that finds criminal *networks*, not just suspicious
> transactions.
>
> FraudLens runs a 3-layer GraphSAGE graph neural network over the Elliptic Bitcoin
> dataset: 203,769 transactions, 234,355 transfers between them, 165 features each.
>
> Here's the thing most fraud models miss. Traditional ML scores each transaction as an
> isolated row in a table. But real fraud is coordinated — rings of accounts cycling money
> between themselves to launder it. Any single hop looks clean. The *pattern* is the crime.
>
> Because the GNN aggregates over each node's 3-hop neighbourhood, it learns something a
> tabular model structurally cannot: a clean-looking wallet surrounded by fraudulent
> wallets is itself suspicious.
>
> What it does:
> → Scores every transaction (AUC 0.9885)
> → Detects coordinated fraud rings via Louvain community detection — found 10, including
>   one that's 97.8% confirmed illicit
> → Explains each flag by backpropagating the prediction into the input features, so you
>   can tell whether a transaction was flagged for its own behaviour or for the company
>   it keeps
>
> The uncomfortable result I decided to publish rather than hide: a plain Random Forest
> beats my GNN on raw F1 (0.93 vs 0.88). Why? Elliptic ships 72 pre-engineered
> neighbourhood-aggregate features, so the tabular model already gets a hand-crafted
> summary of each transaction's graph context for free.
>
> I left that number in the README. The case for the graph model isn't the F1 — it's the
> three things the forest cannot do at any F1: cluster rings, attribute a prediction, and
> generalise to transactions it never saw in training.
>
> Stack: PyTorch Geometric · FastAPI · React + TypeScript · NetworkX · scikit-learn
>
> The console runs with no dataset and no backend — it ships precomputed artifacts so you
> can click through it straight from a clone.
>
> Code: <repo-link>
>
> #MachineLearning #GraphNeuralNetworks #FraudDetection #PyTorch #DataScience

---

## Version B — the narrative one

> "This transaction looks fine."
>
> That sentence is why money laundering works.
>
> I spent the last few weeks building FraudLens, and the thing that stuck with me is how
> badly the standard approach is set up to fail. Score each transaction on its own
> attributes and a laundering ring is invisible by construction — every individual hop is
> designed to look ordinary. The fraud only exists in the shape of the network.
>
> So I stopped treating transactions as rows and treated them as a graph: 203,769 Bitcoin
> transactions as nodes, 234,355 transfers as edges, and a GraphSAGE neural network that
> learns from each node's 3-hop neighbourhood.
>
> It finds the rings. 10 coordinated communities, one of them 97.8% confirmed illicit —
> and for every flag it can tell you whether the verdict came from the transaction's own
> behaviour or from its neighbours.
>
> The part I didn't expect: a basic Random Forest scored *higher* on F1. It took me a
> while to understand why — the dataset ships 72 pre-computed neighbourhood features, so
> the simple model was quietly getting graph context handed to it.
>
> I kept that number in the README instead of tuning until the graph model won. The honest
> comparison is more interesting than a flattering one, and the argument for graphs was
> never about F1 — it's that a tabular model can give you a score and nothing else. No
> rings. No explanation. No generalisation to accounts it hasn't seen.
>
> Built with PyTorch Geometric, FastAPI and React. Clone it and the console runs
> immediately — no dataset download, no backend.
>
> <repo-link>
>
> #GraphNeuralNetworks #FraudDetection #MachineLearning #OpenSource

---

## Notes before you post

- **Swap in the real numbers.** The AUC 0.9885 / F1 0.8833 figures above are from your
  earlier training run. Re-run `fraudlens train` on the restored dataset and check
  `artifacts/metrics.json` before publishing, so the post matches the repo.
- **Carousel order**, if you use one: explorer → rings → explain → model → overview.
- **Don't screenshot the sample-data banner.** Generate real artifacts with
  `fraudlens export` first, or crop above it.
- The "10 fraud rings / 97.8% illicit" figures also come from the real dataset run —
  confirm them the same way.
