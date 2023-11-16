import ivy

class LogisticRegression:

    @staticmethod
    def pred_transform(x):
        if False:
            print('Hello World!')
        return ivy.sigmoid(x)

    @staticmethod
    def first_order_gradient(predt, label):
        if False:
            return 10
        return predt - label

    @staticmethod
    def second_order_gradient(predt, label):
        if False:
            return 10
        return ivy.fmax(predt * (1.0 - predt), 1e-16)

    @staticmethod
    def prob_to_margin(base_score):
        if False:
            print('Hello World!')
        return ivy.logit(base_score)