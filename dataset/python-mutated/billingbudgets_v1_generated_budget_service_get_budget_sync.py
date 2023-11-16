from google.cloud.billing import budgets_v1

def sample_get_budget():
    if False:
        for i in range(10):
            print('nop')
    client = budgets_v1.BudgetServiceClient()
    request = budgets_v1.GetBudgetRequest(name='name_value')
    response = client.get_budget(request=request)
    print(response)