from tinygrad import nn, Tensor, dtypes

# allow for monkeypatching
Embedding = nn.Embedding
Linear = nn.Linear
LayerNorm = nn.LayerNorm

class Bert:
  def __init__(self, hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers, vocab_size, attention_probs_dropout_prob, hidden_dropout_prob):
    CheckSize = hidden_size / num_attention_heads
    assert CheckSize.is_integer() == True, 'The hidden size must be fully divisible by attention heads!'
    self.embeddings = BertEmbeddings(hidden_size, max_position_embeddings, vocab_size, hidden_dropout_prob)
    self.encoder = BertEncoder(hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, attention_probs_dropout_prob, hidden_dropout_prob)

  def __call__(self, input_ids):
    embedding_output = self.embeddings(input_ids)
    encoder_outputs = self.encoder(embedding_output)

    return encoder_outputs

class BertSeq2Seq:
  def __init__(self, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, attention_probs_dropout_prob, hidden_dropout_prob):
    CheckSize = hidden_size / num_attention_heads
    assert CheckSize.is_integer() == True, 'The hidden size must be fully divisible by attention heads!'
    self.encoder = BertEncoder(hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, attention_probs_dropout_prob, hidden_dropout_prob)

  def __call__(self, input_ids):
    encoder_outputs = self.encoder(input_ids)
    return encoder_outputs

class BertEmbeddings:
  def __init__(self, hidden_size, max_position_embeddings, vocab_size,  hidden_dropout_prob):
    self.word_embeddings = Embedding(vocab_size, hidden_size)
    self.position_embeddings = Embedding(max_position_embeddings, hidden_size)
    self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
    self.dropout = hidden_dropout_prob

  def __call__(self, input_ids):
    input_shape = input_ids.shape
    seq_length = input_shape[1]

    position_ids = Tensor.arange(seq_length, requires_grad=False, device=input_ids.device).unsqueeze(0).expand(*input_shape)
    words_embeddings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)

    embeddings = words_embeddings + position_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = embeddings.dropout(self.dropout)
    return embeddings

class BertEncoder:
  def __init__(self, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, attention_probs_dropout_prob, hidden_dropout_prob):
    self.layer = [BertLayer(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob) for _ in range(num_hidden_layers)]

  def __call__(self, hidden_states):
    for layer in self.layer:
      hidden_states = layer(hidden_states)
    return hidden_states

class BertLayer:
  def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
    self.attention = BertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
    self.intermediate = BertIntermediate(hidden_size, intermediate_size)
    self.output = BertOutput(hidden_size, intermediate_size, hidden_dropout_prob)

  def __call__(self, hidden_states):
    attention_output = self.attention(hidden_states)
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output

class BertOutput:
  def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
    self.dense = Linear(intermediate_size, hidden_size)
    self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
    self.dropout = hidden_dropout_prob

  def __call__(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = hidden_states.dropout(self.dropout)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

def gelu(x):
  return x * 0.5 * (1.0 + erf(x / 1.41421))

# approximation of the error function
def erf(x):
  t = (1 + 0.3275911 * x.abs()).reciprocal()
  return x.sign() * (1 - ((((1.061405429 * t + -1.453152027) * t + 1.421413741) * t + -0.284496736) * t + 0.254829592) * t * (-(x.square())).exp())

class BertIntermediate:
  def __init__(self, hidden_size, intermediate_size):
    self.dense = Linear(hidden_size, intermediate_size)

  def __call__(self, hidden_states):
    x = self.dense(hidden_states)
    # tinygrad gelu is openai gelu but we need the original bert gelu
    return gelu(x)

class BertAttention:
  def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
    self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
    self.output = BertSelfOutput(hidden_size, hidden_dropout_prob)

  def __call__(self, hidden_states):
    self_output = self.self(hidden_states)
    attention_output = self.output(self_output, hidden_states)
    return attention_output

class BertSelfAttention:
  def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
    self.num_attention_heads = num_attention_heads
    self.attention_head_size = int(hidden_size / num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = Linear(hidden_size, self.all_head_size)
    self.key = Linear(hidden_size, self.all_head_size)
    self.value = Linear(hidden_size, self.all_head_size)
    self.norm = nn.LayerNorm(hidden_size)
    self.dropout = attention_probs_dropout_prob
    self.scale = self.all_head_size ** -0.5

  def __call__(self, hidden_state):
    hidden_state = self.norm(hidden_state)
    query = self.transpose_for_scores(self.query(hidden_state))
    key = self.transpose_for_scores(self.key(hidden_state))
    value = self.transpose_for_scores(self.value(hidden_state))

    dots = query.matmul(key.transpose(-1,-1)) * self.scale
    attn = dots.softmax(axis=-1)
    attn = attn.dropout(self.dropout)

    context_layer = attn.matmul(value)
    context_layer = context_layer.transpose(1, 2)
    context_layer = context_layer.reshape(context_layer.shape[0], context_layer.shape[1], self.all_head_size)
    return context_layer

  def transpose_for_scores(self, x):
    x = x.reshape(x.shape[0], x.shape[1], self.num_attention_heads, self.attention_head_size)
    return x.transpose(1, 2)

class BertSelfOutput:
  def __init__(self, hidden_size, hidden_dropout_prob):
    self.dense = Linear(hidden_size, hidden_size)
    self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
    self.dropout = hidden_dropout_prob

  def __call__(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = hidden_states.dropout(self.dropout)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

class ViT_bert:
    def __init__(self, channels:int, ImageSize:int, PatchSize:int, numClasses:int, hiddenSize:int, feedforwardDim:int, numAttentionHeads:int, numLayers:int, attentionDropout:float, hiddenDropout:float):
        self.channels = channels
        self.image_size = ImageSize
        self.patch_size = PatchSize
        self.bert = BertSeq2Seq(hidden_size=hiddenSize, intermediate_size=feedforwardDim, num_attention_heads=numAttentionHeads, num_hidden_layers=numLayers, attention_probs_dropout_prob=attentionDropout, hidden_dropout_prob=hiddenDropout)
        self.patch_dim = hiddenSize
        self.SeqLen = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)
        self.EmbedImg = nn.Conv2d(self.channels, self.patch_dim, self.patch_size, self.patch_size)

        # Positional Encoding
        self.position_embeddings = Tensor.zeros(1, self.SeqLen + 1, self.patch_dim)  # +1 for class token
        self.position_embeddings.requires_grad = True

        self.norm = nn.LayerNorm(self.patch_dim)
        self.fc = nn.Linear(hiddenSize, numClasses)

    def __call__(self, img):
        Embed = self.EmbedImg(img).view(img.shape[0], self.SeqLen, self.patch_dim)
        Embed = self.norm(Embed + self.position_embeddings)  # Add positional encoding
        logits = self.bert(Embed)
        logits = logits.mean(axis=1)  # Mean pooling over sequence
        logits = self.fc(logits)
        return logits

class ViT_bertCLS:
    def __init__(self, channels:int, ImageSize:int, PatchSize:int, numClasses:int, hiddenSize:int, feedforwardDim:int, numAttentionHeads:int, numLayers:int, attentionDropout:float, hiddenDropout:float):
        self.channels = channels
        self.image_size = ImageSize
        self.patch_size = PatchSize
        self.bert = BertSeq2Seq(hidden_size=hiddenSize, intermediate_size=feedforwardDim, num_attention_heads=numAttentionHeads, num_hidden_layers=numLayers, attention_probs_dropout_prob=attentionDropout, hidden_dropout_prob=hiddenDropout)
        self.patch_dim = hiddenSize
        self.SeqLen = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)

        # Embedding layers
        self.EmbedImg = nn.Conv2d(self.channels, self.patch_dim, self.patch_size, self.patch_size)

        # Learnable class token
        self.class_token = Tensor.zeros(1,1,self.patch_size)
        self.class_token.requires_grad = True

        # Positional encoding for the patches + class token
        self.position_embeddings = Tensor.zeros(1, self.SeqLen + 1, self.patch_dim)  # +1 for class token
        self.position_embeddings.requires_grad = True

        self.norm = nn.LayerNorm(self.patch_dim)
        self.fc = nn.Linear(hiddenSize, numClasses)

    def __call__(self, img):
        # Step 1: Patch embedding
        Embed = self.EmbedImg(img).view(img.shape[0], self.SeqLen, self.patch_dim)  # [B, SeqLen, patch_dim]

        # Step 2: Add the class token to the beginning of the sequence
        class_token = self.class_token.expand(img.shape[0], -1, -1)  # [B, 1, patch_dim]
        Embed = class_token.cat(Embed, dim=1) # Prepend class token [B, SeqLen+1, patch_dim]

        # Step 3: Add positional encoding
        Embed = Embed + self.position_embeddings

        # Step 4: Apply layer normalization
        Embed = self.norm(Embed)

        # Step 5: Forward pass through the transformer
        logits = self.bert(Embed)

        # Step 6: Use the output corresponding to the class token
        class_logits = logits[:, 0, :]  # Select the first token (class token)

        # Step 7: Final classification layer
        class_logits = self.fc(class_logits)

        return class_logits

if __name__ == '__main__':
    model = ViT_bert(channels=3, ImageSize=224, PatchSize=16, numClasses=10, hiddenSize=384, feedforwardDim=2048, numAttentionHeads=8, numLayers=4, attentionDropout=0.1, hiddenDropout=0.0)
    data = Tensor.rand(1,3,224,224, dtype=dtypes.float32)
    out = model(data).numpy()
    print(out)
