class gat_layer(nn.Module):
    def __init__(self, num_nodes: int, dim_proj: int, num_att_heads: int): 
        super().__init__()

        self.source_dim = 0;
        self.target_dim = 1;

        self.dim_proj = dim_proj
        self.num_nodes = num_nodes
        self.num_att_heads = num_att_heads

        self.source_scores_fn = mx.array([1, num_att_heads, dim_proj]) 
        self.target_scores_fn = mx.array([1, num_att_heads, dim_proj]) 

        self.leakyReLU = nn.LeakyReLU(0.02)

    def __call__(self, node_proj, adjacency_matrix):

        source_idx = adjacency_matrix[self.source_dim]
        target_idx = adjacency_matrix[self.target_dim]

        node_proj = node_proj.reshape([-1, self.num_att_heads, self.dim_proj])
    
        source_scores = (node_proj * self.source_scores_fn).sum(dim=-1)
        target_scores = (node_proj * self.target_scores_fn).sum(dim=-1)

        edge_filtered_node_proj = mx.take(node_proj, source_idx, axis=0);
        edge_filtered_source_scores = mx.take(source_scores, source_idx, axis=0);
        edge_filtered_target_scores = mx.take(target_scores, target_idx, axis=0);

        edge_scores = self.leakyReLU(edge_filtered_source_scores + edge_filtered_target_scores)
        edge_scores = (edge_scores - edge_scores.max()).exp()

        softmax_denominator = mx.zeros([self.num_nodes, self.num_att_heads, self.dim_proj])
        softmax_denominator = softmax_denominator.at[target_idx].add(edge_scores)

        attention_scores = edge_scores / (softmax_denominator + 1e-16)

        edge_filtered_node_proj = edge_filtered_node_proj * attention_scores;

        new_node_proj = mx.zeros([self.num_nodes, self.num_att_heads, self.dim_proj]).at(target_idx).add(edge_filtered_node_proj)
        new_node_proj = self.leakyReLU(new_node_proj)
        new_node_proj = new_node_proj.reshape((self.num_nodes, self.num_att_heads * self.dim_proj))

        return new_node_proj
