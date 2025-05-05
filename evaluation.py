# Few-shot evaluation
def evaluate_few_shot(model, task_sampler):
    accuracies = []
    for task in task_sampler:
        support_logits = model(task.support)
        query_logits = model(task.query)
        acc = compute_accuracy(support_logits, query_logits)
        accuracies.append(acc)
    return np.mean(accuracies)

# Backward transfer evaluation
base_test_accuracy = evaluate_standard(model, base_test_loader)