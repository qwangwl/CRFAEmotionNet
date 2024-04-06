
class EarlyStopping(object):
    def __init__(self, patience=5):
        # a simple early stopping
        super(EarlyStopping, self).__init__()
        
        self.patience = patience
        self.best_score = None
        self.early_stop = False
        
        self.counter = 0
        
    def __call__(self, acc):
        
        if self.best_score is None:
            self.best_score = acc
            
        elif acc <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = acc
            self.counter = 0