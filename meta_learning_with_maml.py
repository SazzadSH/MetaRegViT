from torchmeta.modules import MetaModule
from torchmeta.utils.gradient_based import maml_update

# Convert model to MAML-compatible
maml_model = MetaModule(MetaRegViT(vit_model))

# 5-way 1/5-shot tasks
task_sampler = torchmeta.utils.data.TaskSampler(
    dataset=base_dataset, ways=5, shots=5, test_shots=15
)

for meta_epoch in range(100):
    for task_batch in task_sampler:
        # Inner loop adaptation
        learner = maml_model.clone()
        adaptation_loss = 0
        for images, labels in task_batch.support:
            adaptation_loss += learner(images, labels)
        learner.adapt(adaptation_loss)
        
        # Outer loop update
        evaluation_loss = learner(task_batch.query)
        evaluation_loss.backward()
        optimizer.step()