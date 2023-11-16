from google.cloud.billing import budgets_v1beta1

def sample_get_budget():
    if False:
        while True:
            i = 10
    client = budgets_v1beta1.BudgetServiceClient()
    request = budgets_v1beta1.GetBudgetRequest(name='name_value')
    response = client.get_budget(request=request)
    print(response)