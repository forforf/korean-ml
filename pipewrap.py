from sklearn.pipeline import Pipeline


#A simple wrapper around a pipeline that allows for comparing pipelines after serialization
class PipeWrap:

    def __init__(self, steps):
        self.params = [step[1].get_params() for step in steps]
        self.steps = steps
        self.pipe = Pipeline(steps=self.steps)

    def predict(self, X):
        return self.pipe.predict(X)
