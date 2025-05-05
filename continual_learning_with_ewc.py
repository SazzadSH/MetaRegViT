# Elastic Weight Consolidation (EWC) for each new task
def compute_fisher_matrix(model, dataset):
    # (Compute Fisher Information for important parameters)
    return fisher_matrix

for task_name, task_data in incremental_tasks.items():
    fisher_matrix = compute_fisher_matrix(model, task_data)
    
    # Train on new task with EWC regularization
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for epoch in range(10):
        for images, labels in task_data:
            outputs = model(images, task_id=task_name)
            loss = loss_fn(outputs, labels) + ewc_lambda * fisher_penalty
            loss.backward()
            optimizer.step()