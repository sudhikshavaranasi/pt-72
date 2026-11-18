"""Five binary candidate classifiers with common fold-fitted preprocessing."""
import numpy as np

CONFIGS = {
    "logistic": dict(l2=.003, steps=100),
    "gaussian_nb": dict(floor=.03, temperature=1.0),
    "rbf_prototypes": dict(bandwidth=.9, centers=10),
    "boosted_stumps": dict(rate=.16, rounds=60, bins=20),
    "shallow_mlp": dict(rate=.006, hidden=24, epochs=7),
}
SPACES = {
    "logistic": (("l2", .0001, .05, False), ("steps", 70, 200, True)),
    "gaussian_nb": (("floor", .002, .2, False), ("temperature", .4, 2., False)),
    "rbf_prototypes": (("bandwidth", .3, 2., False), ("centers", 6, 22, True)),
    "boosted_stumps": (("rate", .05, .35, False), ("rounds", 40, 125, True)),
    "shallow_mlp": (("rate", .001, .02, False), ("hidden", 12, 44, True)),
}


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def sqdist(x, centers):
    return np.maximum((x*x).sum(1)[:, None] + (centers*centers).sum(1) - 2*(x @ centers.T), 0)


class CandidateModel:
    def __init__(self, name, feature_cols, config, seed):
        self.name = name
        self.feature_cols = tuple(feature_cols)
        self.config = dict(config)
        self.seed = seed

    def fit(self, x, y, w):
        self.center = np.nanmedian(x, axis=0)
        self.scale = np.maximum(np.where(np.isfinite(x), x, self.center).std(axis=0), 1e-5)
        x = self.transform(x)
        p = self.config
        rng = np.random.default_rng(self.seed)
        prior = np.average(y, weights=w)
        self.intercept = float(np.log(prior / (1-prior)))
        if self.name == "logistic":
            self.coef = np.zeros(x.shape[1], dtype=np.float32)
            for _ in range(p["steps"]):
                error = (sigmoid(x @ self.coef + self.intercept) - y) * w / w.sum()
                self.coef -= .2 * (x.T @ error + p["l2"] * self.coef)
                self.intercept -= .2 * error.sum()
        elif self.name == "gaussian_nb":
            self.gaussians = []
            for c in (0, 1):
                use = y == c
                mean = np.average(x[use], axis=0, weights=w[use])
                var = np.average((x[use]-mean)**2, axis=0, weights=w[use])
                self.gaussians.append((mean, var + p["floor"]))
        elif self.name == "rbf_prototypes":
            self.prototypes = []
            for c in (0, 1):
                rows = np.flatnonzero(y == c)
                rows = rng.choice(rows, min(9000, len(rows)), replace=False)
                a, weights = x[rows], w[rows]
                centers = a[rng.choice(len(a), p["centers"], replace=False)].copy()
                for _ in range(9):
                    group = sqdist(a, centers).argmin(axis=1)
                    for j in range(len(centers)):
                        use = group == j
                        if use.any():
                            centers[j] = np.average(a[use], axis=0, weights=weights[use])
                group = sqdist(a, centers).argmin(axis=1)
                mass = np.bincount(group, weights=weights, minlength=len(centers)) + 1e-8
                self.prototypes.append((centers, mass / mass.sum()))
        elif self.name == "boosted_stumps":
            self.fit_stumps(x, y, w)
        elif self.name == "shallow_mlp":
            self.fit_mlp(x, y, w, rng)
        else:
            raise ValueError(self.name)
        return self

    def transform(self, x):
        return np.clip((np.where(np.isfinite(x), x, self.center)-self.center)/self.scale, -15, 15)

    def fit_stumps(self, x, y, w):
        p = self.config
        cuts = [np.unique(np.quantile(c, np.linspace(0,1,p["bins"] + 1)[1:-1])) for c in x.T]
        bins = [np.searchsorted(k, c, side="left") for k, c in zip(cuts, x.T)]
        pred = np.full(len(x), self.intercept)
        self.stumps = []
        for _ in range(p["rounds"]):
            score = sigmoid(pred)
            grad, hess = w * (y-score), w * score * (1-score)
            best = None
            for feature, (column, thresholds) in enumerate(zip(bins, cuts)):
                if not len(thresholds):
                    continue
                g = np.bincount(column, weights=grad, minlength=len(thresholds)+1).cumsum()
                h = np.bincount(column, weights=hess, minlength=len(thresholds)+1).cumsum()
                left, right = h[:-1]+2, h[-1]-h[:-1]+2
                gain = g[:-1]**2/left + (g[-1]-g[:-1])**2/right
                j = int(gain.argmax())
                choice = (gain[j], feature, thresholds[j], p["rate"]*g[j]/left[j],
                          p["rate"]*(g[-1]-g[j])/right[j])
                if best is None or choice[0] > best[0]:
                    best = choice
            if best is None:
                break
            _, feature, threshold, left, right = best
            self.stumps.append((feature, threshold, left, right))
            pred += np.where(x[:,feature] <= threshold, left, right)

    def fit_mlp(self, x, y, w, rng):
        p = self.config
        h = p["hidden"]
        params = [rng.normal(0,.3,(x.shape[1],h)).astype(np.float32), np.zeros(h,np.float32),
                  rng.normal(0,.2,h).astype(np.float32), np.array(self.intercept,np.float32)]
        first = [np.zeros_like(v) for v in params]
        second = [np.zeros_like(v) for v in params]
        step = 0
        for _ in range(p["epochs"]):
            order = rng.permutation(len(y))
            for start in range(0, len(y), 2048):
                step += 1
                i = order[start:start+2048]
                a = x[i]
                h = np.tanh(a @ params[0] + params[1])
                e = (sigmoid(h @ params[2] + params[3])-y[i])*w[i]/w[i].sum()
                d = e[:,None]*params[2]*(1-h*h)
                for j,g in enumerate((a.T @ d, d.sum(0), h.T @ e, e.sum())):
                    first[j] = .9*first[j] + .1*g
                    second[j] = .999*second[j] + .001*g*g
                    params[j] -= p["rate"] * first[j]/(1-.9**step)/(np.sqrt(second[j]/(1-.999**step))+1e-7)
        self.params = params

    def decision(self, raw):
        x = self.transform(raw)
        p = self.config
        if self.name == "logistic":
            return x @ self.coef + self.intercept
        if self.name == "gaussian_nb":
            a,b = [-.5*(np.log(v)+(x-m)**2/v).sum(1) for m,v in self.gaussians]
            return (b-a+self.intercept)/p["temperature"]
        if self.name == "rbf_prototypes":
            values = []
            for centers,mass in self.prototypes:
                a = np.log(mass) - sqdist(x,centers)/(2*p["bandwidth"]**2)
                top = a.max(1)
                values.append(top + np.log(np.exp(a-top[:,None]).sum(1)))
            return values[1]-values[0]+self.intercept
        if self.name == "boosted_stumps":
            out = np.full(len(x),self.intercept)
            for feature, threshold, left, right in self.stumps:
                out += np.where(x[:,feature] <= threshold,left,right)
            return out
        weights,bias,output,final_bias = self.params
        return np.tanh(x @ weights+bias) @ output+final_bias

    def probability(self, x):
        out = np.empty(len(x), np.float32)
        for i in range(0,len(x),32768):
            out[i:i+32768] = sigmoid(self.decision(x[i:i+32768]))
        return out
