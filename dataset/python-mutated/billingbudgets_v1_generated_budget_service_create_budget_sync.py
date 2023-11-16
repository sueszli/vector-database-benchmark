from google.cloud.billing import budgets_v1

def sample_create_budget():
    if False:
        for i in range(10):
            print('nop')
    client = budgets_v1.BudgetServiceClient()
    request = budgets_v1.CreateBudgetRequest(parent='parent_value')
    response = client.create_budget(request=request)
    print(response)