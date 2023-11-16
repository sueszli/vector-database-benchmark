from google.cloud.billing import budgets_v1

def sample_list_budgets():
    if False:
        for i in range(10):
            print('nop')
    client = budgets_v1.BudgetServiceClient()
    request = budgets_v1.ListBudgetsRequest(parent='parent_value')
    page_result = client.list_budgets(request=request)
    for response in page_result:
        print(response)