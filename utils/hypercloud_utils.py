import torch 

# TODO: Implement data mocks to verify how model learns when it needs to find only certain group of values
def mock_gs_parameters():
    return {
        "rgb": None,
        "c": None,
        "alphas": None,
        "opacity": None,
    }

def create_embeddings_for_faces(faces, sphere_faces_indices, get_embedder_fn):
    faces_with_indexes = torch.column_stack([faces, sphere_faces_indices])
    embed_fn, _ = get_embedder_fn(4, 0)

    out = embed_fn(faces_with_indexes)
    return out