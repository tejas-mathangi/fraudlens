# LinkedIn post drafts

Two versions. Pick one, don't post both. Lead with
`docs/screenshots/rings.png` — the ring table beside a dense 100%-illicit cluster is the
single clearest image in the project, and it survives feed thumbnailing.

Replace `<repo-link>` with `https://github.com/tejas-mathangi/fraudlens`.

**Every number below is verified** against `artifacts/metrics.json` from a real run on
the full Elliptic dataset, not estimated.

---

## Version A — the technical one

> I built a fraud detection system that finds criminal *networks*, not just suspicious
> transactions.
>
> FraudLens runs a 3-layer GraphSAGE graph neural network over the Elliptic Bitcoin
> dataset: 203,769 transactions, 234,355 transfers between them, 165 features each.
>
> Here's what most fraud models miss. Traditional ML scores each transaction as an
> isolated row in a table. But real fraud is coordinated — rings of accounts cycling
> money between themselves to launder it. Any single hop looks clean. The *pattern* is
> the crime.
>
> Because the GNN aggregates over each node's 3-hop neighbourhood, it learns something a
> tabular model structurally cannot: a clean-looking wallet surrounded by fraudulent
> wallets is itself suspicious.
>
> What it does:
> → Scores every transaction (AUC 0.9885)
> → Detects coordinated fraud rings via Louvain community detection — found 10, one of
>   them 39 accounts where 100% of the labelled members are confirmed illicit
> → Explains each flag by backpropagating the prediction into the input features, so you
>   can tell whether a transaction was flagged for its own behaviour or for the company
>   it keeps
>
> Now the part I nearly got wrong.
>
> A plain Random Forest beats my GNN on F1: 0.935 vs 0.883. My first instinct was to
> bury that. Then I looked at the confusion matrices instead of the summary metric:
>
> Random Forest — caught 803 frauds, missed 106, 5 false alarms
> GraphSAGE — caught 840 frauds, missed 69, 153 false alarms
>
> The GNN misses 35% fewer frauds. It just raises more false alarms doing it. Which
> model is "better" depends entirely on whether a missed fraud costs you more than a
> false alarm — and in anti-money-laundering it usually does. That asymmetry is exactly
> what the 5.1× class weighting in the loss function encodes.
>
> The Random Forest also scores well for a specific reason: Elliptic ships 72
> pre-engineered neighbourhood-aggregate features, so the tabular model is quietly being
> handed a hand-crafted summary of each transaction's graph context.
>
> Both numbers are in the README. Reporting the baseline honestly was more interesting
> than tuning until the graph model won.
>
> Stack: PyTorch Geometric · FastAPI · React + TypeScript · NetworkX · scikit-learn
>
> The console runs with no dataset and no backend — the repo ships precomputed artifacts,
> so you can click through it straight from a clone.
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
> I spent the last few weeks building FraudLens, and what stuck with me is how badly the
> standard approach is set up to fail. Score each transaction on its own attributes and a
> laundering ring is invisible by construction — every individual hop is *designed* to
> look ordinary. The fraud only exists in the shape of the network.
>
> So I stopped treating transactions as rows and treated them as a graph: 203,769 Bitcoin
> transactions as nodes, 234,355 transfers as edges, and a GraphSAGE neural network that
> learns from each node's 3-hop neighbourhood.
>
> It finds the rings. 10 coordinated communities covering 382 confirmed-illicit
> transactions — including one cluster of 39 accounts where every single labelled member
> is confirmed illicit. And for each flag it can say whether the verdict came from the
> transaction's own behaviour or from its neighbours.
>
> The part I didn't expect: a basic Random Forest scored higher on F1. 0.935 to 0.883.
>
> I almost tuned the GNN until it won. Instead I looked at what the two models actually
> did, and the ranking flipped — the GNN caught 840 frauds to the forest's 803, and missed
> 69 where the forest missed 106. It raises more false alarms to get there. In fraud
> detection that's usually the trade you want: a false alarm costs an analyst ten minutes,
> a missed fraud costs real money.
>
> The forest also had help I hadn't accounted for. The dataset ships 72 pre-computed
> neighbourhood features, so the "simple" model was being handed graph context for free.
>
> Both numbers stayed in the README. The honest comparison taught me more than a
> flattering one would have, and the argument for graphs was never about F1 anyway — a
> tabular model can give you a score and nothing else. No rings. No explanation. No
> generalisation to accounts it has never seen.
>
> Built with PyTorch Geometric, FastAPI and React. Clone it and the console runs
> immediately — no dataset download, no backend.
>
> <repo-link>
>
> #GraphNeuralNetworks #FraudDetection #MachineLearning #OpenSource

---

## Notes before you post

- **Carousel order**, if you use one: rings → explorer → explain → model → overview.
- The screenshots in `docs/screenshots/` are from the real dataset — no sample-data
  banner to crop.
- If you retrain, re-check the figures: `fraudlens evaluate` refreshes
  `artifacts/metrics.json` from the saved checkpoint without touching it.
- Version A leads with capability, Version B with the mistake-and-correction arc. B tends
  to travel further; A reads as more senior.
