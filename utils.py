import torch
import torch.nn.functional as F

# Define similarity function (cosine similarity)
def similarity(h1, h2):
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)
    return torch.matmul(h1, h2.T)

# Loss function for mixed graphs (L_mix)
def mix_loss(h_mix_list, h_graph, batch_size, K):
    """
    h_mix_list: List of Tensors of shape (B, D), each for a different bias environment.
    h_graph: Tensor of shape (B, D), original graph embeddings.
    batch_size: Number of graphs in a batch.
    K: Number of bias environments.
    """
    loss = 0
    for i in range(batch_size):
        for j in range(K):
            # Positive sample: h_mix_list[j][i] and h_graph[i]
            pos_sim = similarity(h_mix_list[j][i:i+1], h_graph[i:i+1])

            # Negative samples: h_mix_list[j][i] and all other h_graph[k] (k != i)
            neg_sims = similarity(h_mix_list[j][i:i+1], h_graph)
            neg_sims[i] = -float('inf')  # Mask self similarity

            # InfoNCE loss for this environment
            loss += -torch.log(torch.exp(pos_sim) / torch.exp(neg_sims).sum())

    return loss / (batch_size * K)

# Loss function for environment consistency (L_env)
def env_loss(h_mix_list, batch_size, K):
    """
    h_mix_list: List of Tensors of shape (B, D), each for a different bias environment.
    batch_size: Number of graphs in a batch.
    K: Number of bias environments.
    """
    loss = 0
    for i in range(batch_size):
        for j in range(K):
            for l in range(K):
                if j != l:
                    loss += similarity(h_mix_list[j][i:i+1], h_mix_list[l][i:i+1])

    return loss / (batch_size * K * (K - 1))

# Total loss function
def total_loss(h_graph, h_mix_list, labels, model, lambda_mix=1.0, lambda_env=1.0):
    """
    h_graph: Tensor of shape (B, D), original graph embeddings
    h_mix_list: List of Tensors, each of shape (B, D), embeddings of mixed graphs
    labels: Tensor of shape (B,), ground truth labels
    model: Prediction model to compute regression loss
    """
    batch_size = h_graph.size(0)
    K = len(h_mix_list)

    # Regression loss (MSE)
    predictions = model(h_graph)
    regression_loss = F.mse_loss(predictions, labels)

    # Mix loss
    mix_loss_value = mix_loss(h_mix_list, h_graph, batch_size, K)

    # Environment consistency loss
    env_loss_value = env_loss(h_mix_list, batch_size, K)

    # Total loss
    total = regression_loss + lambda_mix * mix_loss_value + lambda_env * env_loss_value
    return total

# Example usage
if __name__ == "__main__":
    # Simulated example: batch size B=4, embedding dimension D=16, number of bias environments K=3
    B, D, K = 4, 16, 3

    # Random embeddings for original graphs (h_graph), mixed graphs (h_mix_list), and labels
    h_graph = torch.rand(B, D)
    h_mix_list = [torch.rand(B, D) for _ in range(K)]
    labels = torch.rand(B)

    # Dummy prediction model
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=-1)  # Sum embeddings as dummy prediction

    model = DummyModel()

    # Compute loss
    loss = total_loss(h_graph, h_mix_list, labels, model, lambda_mix=1.0, lambda_env=1.0)
    print("Total loss:", loss.item())
