[Image retrieval](/learn/multimodal-rag/image-retrieval) treated an image encoder as a black box: a picture goes in, a 512-dimensional vector comes out, and pictures that look alike get nearby vectors. That was enough to build a search, and also enough to be surprised by it: a new screenshot of a known error ranked below an unrelated screen. This lesson opens the box. You will follow an image through a vision transformer step by step, from pixels to patches to tokens to one vector; see that what "alike" means is decided by how the encoder was trained; and measure, on real embeddings, which changes to an image the vector ignores and which it notices.

The goal is practical judgement: knowing which encoder family suits a corpus, which preprocessing step silently throws away content, and how to read a cosine score from an image model.

## Learning objectives

- Trace an image through a vision transformer, naming the shape of the data after each step from pixel tensor to normalised embedding.
- Compute the number of tokens and the attention cost for a given image size and patch size.
- Explain how the training objective (class labels, captions, self-supervision, document pages) determines which similarities an encoder captures, and choose a family for a retrieval task.
- Measure an encoder's invariances on your own images and interpret the results relative to the score of an unrelated image.
- Identify preprocessing that discards content, such as centre-cropping wide images, and replace it.

## Prerequisites

- [Image retrieval](/learn/multimodal-rag/image-retrieval): the visual index and query by example.
- [What are embeddings?](/learn/embedding-fundamentals/what-are-embeddings): vectors whose distances encode similarity.
- [Vector normalisation](/learn/embedding-mathematics/vector-normalisation) and [Cosine similarity](/learn/embedding-mathematics/cosine-similarity).
- [Late interaction](/learn/reranking-models/late-interaction): many vectors per item, scored with MaxSim (useful for the section on pooling).

## Why this matters

When a text embedding misbehaves, you can read the text and usually see why. An image embedding gives you no such handle; it is 512 numbers computed from a million pixels. Teams therefore treat image encoders as interchangeable and their scores as self-explanatory, and three kinds of failure follow.

**The wrong notion of similarity.** An encoder trained to name objects in photographs considers two laptops similar because both are laptops. A support engineer looking for "the same model of laptop, with the same damaged hinge" wants a much finer similarity. An encoder trained without any labels might serve that need better. Picking a model by leaderboard rank, without asking what it was trained to consider similar, picks at random for your task.

**Content discarded before the model sees it.** Most image encoders take a fixed-size square. The standard preprocessing for CLIP resizes the shorter side to 224 pixels and cuts out the central square. For a wide dashboard screenshot, that throws away both sides of the image, and nothing in the output warns you.

**Scores read on the wrong scale.** In this lesson's measurements a floor plan and a VPN error dialog, which have nothing in common, score 0.58 against each other. Anyone expecting cosine near zero for unrelated content sets thresholds that never fire.

Understanding the encoder turns these from mysteries into checks you can run.

## The intuition

Imagine describing a photograph to a friend over the phone in exactly 512 numbers, using a code book you both learned in advance. What the code book can express depends on how you learned it. If you learned it by playing "name the object" with thousands of labelled photos, your numbers will say "laptop, desk, indoor" very precisely and say little about the scratch on the lid. If you learned it by matching photos to their captions from the web, your numbers will capture whatever captions tend to mention. If you learned it by recognising that two cropped, recoloured versions of the same photo are the same photo, your numbers will capture the specific look of that scene.

There is also the matter of how you look before you speak. If the rule is "glance at a square in the middle of the picture, at low resolution", anything in the corners or in fine print never makes it into your description, however good your code book.

A visual embedding is that 512-number description. The training decides the code book; the preprocessing decides what you looked at.

## How it works

### From pixels to one vector: the vision transformer

Most current image encoders used for retrieval are **vision transformers** (ViT, Dosovitskiy and colleagues, 2020): the same transformer architecture used for text, applied to an image cut into squares. Earlier encoders were **convolutional neural networks** (CNNs, such as ResNet), which slide small learned filters across the image and build up features layer by layer; they are still used and work well, but the steps below follow a ViT, since that is what CLIP ViT-B/32 is.

1. **Decode and convert.** Read the file into an RGB array of shape height by width by 3, with values from 0 to 255.
2. **Resize and crop.** Resize so that the shorter side is 224 pixels, then cut the central 224 by 224 square. (This is CLIP's default; other models use other sizes and rules.)
3. **Normalise.** Scale values to 0 to 1, then subtract a per-channel mean and divide by a per-channel standard deviation, the statistics the model was trained with. The result is a tensor of shape 3 by 224 by 224.
4. **Cut into patches.** Divide the square into a grid of non-overlapping patches. With 32-pixel patches, that is 7 by 7 = 49 patches, each holding $32 \times 32 \times 3 = 3{,}072$ numbers.
5. **Embed each patch.** A learned linear map turns each patch's 3,072 numbers into a 768-dimensional vector. A learned **position embedding** is added to each, so the model knows where in the grid the patch came from.
6. **Add a class token.** A learned extra vector, the **class token** (often written `[CLS]`), is put at the front of the sequence. It corresponds to no patch; its job is to collect a summary. The sequence is now 50 tokens of 768 dimensions.
7. **Run the transformer.** Twelve layers of self-attention and feed-forward blocks let every token exchange information with every other. After them, each patch token describes its patch in the context of the whole image, and the class token describes the whole image.
8. **Pool.** Take the class token's final vector (after a layer normalisation) as the image summary: 768 dimensions.
9. **Project and normalise.** A final linear map projects to the shared embedding size, 512 for CLIP ViT-B/32, and the vector is scaled to unit length so that dot products are cosines.

```mermaid
flowchart LR
    A["Image file"] --> B["Resize shorter side<br/>to 224, centre crop"]
    B --> C["Normalise per channel<br/>3 x 224 x 224"]
    C --> D["49 patches<br/>of 32 x 32 x 3"]
    D --> E["Linear patch embedding<br/>+ position embedding"]
    E --> F["Prepend class token<br/>50 x 768"]
    F --> G["12 transformer layers"]
    G --> H["Class token output<br/>768"]
    H --> I["Projection + unit length<br/>512"]
```

Steps 1 to 3 run outside the model, in a separate **image processor** shipped with it. They are easy to overlook and, as you will see, they are where content is most often lost.

### The training objective decides what "similar" means

The architecture says how a vector is computed. The training says what it is *for*. Four families matter for retrieval.

| Family | Trained to | Similar vectors mean | Good for | Examples |
| --- | --- | --- | --- | --- |
| Supervised classifiers | Predict a label from a fixed list (such as the 1,000 ImageNet classes) | Same kind of object | Coarse category search | ResNet, ViT trained on ImageNet |
| Language-supervised (contrastive) | Match images to their captions | Would be described in similar words | Text-to-image search, general photo search | CLIP, SigLIP |
| Self-supervised | Recognise two altered views of one image as the same, without labels | Look alike in structure and detail | Image-to-image search, near-duplicates, fine-grained matching | SimCLR, DINO, DINOv2 |
| Document and page models | Match text queries to rendered document pages | Answer similar questions | Visually rich pages: slides, forms, brochures | ColPali and related models |

A few points sharpen the table.

- **Language supervision gives a vocabulary, and its limits.** CLIP-style models learn the concepts that captions on the web mention. That is what lets them search images with text (the subject of [Image-text matching](/learn/multimodal-rag/image-text-matching)). It also means they learn little about distinctions captions rarely make, such as the difference between two error codes in a dialog.
- **Self-supervised models are often better for image-to-image search.** With no captions involved, nothing pushes them to discard visual detail that has no name. The DINOv2 authors (Oquab and colleagues, 2023) present their features as general-purpose for image-level and pixel-level tasks, including retrieval. They produce no text-compatible vectors, so they cannot serve text queries on their own.
- **Page models are a different tool.** They are trained on screenshots of documents paired with questions, read text in images far better than CLIP, and usually keep one vector per patch. [Multimodal document representation](/learn/multimodal-ingestion/multimodal-document-representation) covers them; they retrieve pages, not photos.

The practical rule: pick the family whose training notion of similarity matches your query, then compare two or three candidates on a sample of your own queries. Published benchmarks measure someone else's notion of similarity.

### Pooling: one vector or many

Step 8 kept only the class token and discarded the 49 patch tokens. That is the standard choice for a single-vector index, and alternatives exist:

- **Mean or attention pooling** combines the patch tokens instead. Some models are trained this way; always use the pooling the model was trained with, since the projection layer expects it.
- **Keep the patch tokens** as a multi-vector representation and score with MaxSim, as [late interaction](/learn/reranking-models/late-interaction) does for text. A query can then match one small region strongly instead of the whole image weakly. The cost is one stored vector per patch: 49 times the storage for ViT-B/32 at 224 pixels, and far more for higher resolutions and smaller patches.

### Resolution, aspect ratio and the centre crop

The fixed input square creates two problems for document images.

**Small detail disappears.** A 640 by 400 screenshot is scaled by $224/400 = 0.56$, so each 32-pixel patch covers about a 57-pixel square of the original: a word or two of body text, blended into one token. Text-heavy images lose their text first.

**Off-centre content is cropped away.** Resize-then-centre-crop keeps only the middle square of a wide or tall image. For a 1280 by 400 dashboard screenshot, the kept region is the middle 400 original pixels; both sides are gone. Even the 640 by 400 screenshots of the previous lesson lost 120 pixels on each side, including the first characters of the error banner. The code below demonstrates it by changing only the right edge of a wide image and getting a cosine of exactly 1.

Remedies, roughly in order of cost:

- **Pad to a square** (letterboxing) before preprocessing, so nothing is cropped. Everything is kept at lower resolution.
- **Tile**: cut the image into overlapping squares and embed each, with a pointer to the parent image, as described in [Image retrieval](/learn/multimodal-rag/image-retrieval).
- **Use a higher-resolution model.** Encoders exist at 336 pixels and above, and some recent models accept variable aspect ratios; check the model card for the exact input handling, because it varies between releases.

### Invariance: what the vector ignores

An encoder is **invariant** to a change if the vector barely moves when the change is applied. Invariances are learned from training data and augmentation: models trained with random crops, colour jitter and blur become insensitive to them, deliberately, so that a photo still matches after recompression or in different lighting. Two consequences follow.

- Invariance is what makes query by example work on phone photos and recompressed copies.
- Invariance is also blindness. If the training taught the model to ignore something, it cannot retrieve by it. A model trained with colour augmentation will not reliably separate "the red error banner" from "the amber warning banner".

The only reliable way to know a model's invariances on *your* images is to measure them, which is what the code does.

## The mathematics

### Normalising a pixel

**Intuition.** The encoder was trained on inputs centred around zero with roughly unit spread in each channel. Feeding it raw 0 to 255 values would put every input far outside what it has seen.

$$
x' = \frac{x / 255 - \mu_c}{\sigma_c}
$$

$x$ is a raw pixel value in channel $c$ (red, green or blue); $\mu_c$ and $\sigma_c$ are that channel's mean and standard deviation over the training images, stored in the model's preprocessor configuration. For CLIP, $\mu = (0.4815, 0.4578, 0.4082)$ and $\sigma = (0.2686, 0.2613, 0.2758)$.

**By hand.** A pure white pixel, $x = 255$, in the blue channel: $(1 - 0.4082) / 0.2758 = 0.5918 / 0.2758 = 2.146$. The same pixel in the red channel: $(1 - 0.4815) / 0.2686 = 1.930$. A dark grey pixel of 30 in red: $(0.1176 - 0.4815) / 0.2686 = -1.355$.

### Tokens and attention cost

**Intuition.** A ViT turns an image into a sequence of tokens, one per patch plus the class token. Self-attention compares every token with every other, so its cost grows with the square of the number of tokens.

$$
n = \left(\frac{S}{P}\right)^2 + 1, \qquad \text{attention pairs per layer} = n^2
$$

$S$ is the side of the input square in pixels and $P$ the patch side. The patch-embedding layer maps $P \times P \times 3$ numbers to a $D$-dimensional token, so it has $3P^2 D$ weights.

**By hand.**

- ViT-B/32 at 224 pixels: $(224/32)^2 + 1 = 49 + 1 = 50$ tokens; $50^2 = 2{,}500$ attention pairs per layer. The patch embedding has $3 \times 32^2 \times 768 = 2{,}359{,}296$ weights.
- ViT-B/16 at 224: $(224/16)^2 + 1 = 197$ tokens; $197^2 = 38{,}809$ pairs, about 15.5 times as many. Smaller patches see finer detail, at a steep price.
- ViT-L/14 at 336: $(336/14)^2 + 1 = 577$ tokens; $577^2 = 332{,}929$ pairs, 133 times the B/32 figure, before counting that L models are also wider and deeper.

```python
for name, S, P in [("B/32 @224", 224, 32), ("B/16 @224", 224, 16), ("L/14 @336", 336, 14)]:
    n = (S // P) ** 2 + 1
    print(f"{name}: {n} tokens, {n * n:,} attention pairs per layer, {n * n / 2500:.1f}x B/32")
print("patch-embedding weights, B/32:", f"{3 * 32 * 32 * 768:,}")
```

```output
B/32 @224: 50 tokens, 2,500 attention pairs per layer, 1.0x B/32
B/16 @224: 197 tokens, 38,809 attention pairs per layer, 15.5x B/32
L/14 @336: 577 tokens, 332,929 attention pairs per layer, 133.2x B/32
patch-embedding weights, B/32: 2,359,296
```

The token count also sets the storage of a multi-vector representation: 49 patch vectors for B/32 at 224, 576 for L/14 at 336.

## In code

### Opening the encoder and measuring its invariances

**The problem.** See the shapes inside a real encoder, confirm that the library's one-line `encode` does exactly those steps, then measure how far the embedding moves under common changes to a screenshot, compared with a naive raw-pixel vector.

**Dependencies.** The same as [Image retrieval](/learn/multimodal-rag/image-retrieval), including its `meridian_images.py` file, which draws the synthetic screenshots. Sentence Transformers wraps the Hugging Face `transformers` CLIP model, which this script reaches into.

```bash
pip install sentence-transformers pillow numpy torch
```

```python
import io
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from sentence_transformers import SentenceTransformer
from meridian_images import COLLECTION, screen

clip = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
hf_model = clip[0].model                         # the underlying transformers CLIPModel
image_processor = clip[0].processor.image_processor

# 1. Follow one image through the encoder.
original = COLLECTION["vpn-e4021"][0]
pixels = image_processor(images=original, return_tensors="pt")["pixel_values"].to(hf_model.device)
with torch.no_grad():
    vision_out = hf_model.vision_model(pixel_values=pixels)
    projected = hf_model.visual_projection(vision_out.pooler_output)
print("pixel tensor:   ", tuple(pixels.shape))
print("token sequence: ", tuple(vision_out.last_hidden_state.shape), "(1 class token + 7 x 7 patches)")
print("pooled, 768-d:  ", tuple(vision_out.pooler_output.shape))
print("projected:      ", tuple(projected.shape))
st_vec = clip.encode([original], normalize_embeddings=True)[0]
manual = torch.nn.functional.normalize(projected, dim=-1)[0].cpu().numpy()
print(f"same as clip.encode? cosine = {float(manual @ st_vec):.4f}\n")

# 2. What survives and what does not: embedding cosine vs raw-pixel cosine.
def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(buf).convert("RGB")

def shift(img, dx):
    out = Image.new("RGB", img.size, "white")
    out.paste(img, (dx, 0))
    return out

def pixel_vec(img):
    v = np.asarray(img.convert("L").resize((64, 40)), dtype=np.float32).ravel()
    v -= v.mean()
    return v / np.linalg.norm(v)

variants = {
    "half size":                 original.resize((320, 200)),
    "JPEG quality 20":           jpeg(original, 20),
    "Gaussian blur":             original.filter(ImageFilter.GaussianBlur(2)),
    "greyscale":                 ImageOps.grayscale(original).convert("RGB"),
    "shifted 40 px right":       shift(original, 40),
    "rotated 10 degrees":        original.rotate(10, fillcolor="white"),
    "colours inverted":          ImageOps.invert(original),
    "other error (E-4012)":      COLLECTION["vpn-e4012"][0],
    "other app, same layout":    screen("Meridian Reports", "Export failed: report too large",
                                        ["Reduce the date range.", "Then try again."]),
    "unrelated (floor plan)":    COLLECTION["floor-plan"][0],
}

names = list(variants)
embs = clip.encode([variants[n] for n in names], normalize_embeddings=True)
p0 = pixel_vec(original)
print(f"{'variant':26s} {'embedding':>9s} {'pixels':>7s}")
for n, e in zip(names, embs):
    print(f"{n:26s} {float(e @ st_vec):9.3f} {float(pixel_vec(variants[n]) @ p0):7.3f}")

# 3. The centre crop: a wide image loses its sides before it is embedded.
wide = Image.new("RGB", (1280, 400), "white")
wide.paste(original, (320, 0))                   # original in the middle
wide_e4012 = wide.copy()
wide_e4012.paste(screen("Meridian VPN", "Error E-4012: certificate expired"), (960, 0))  # change only the right edge
a, b = clip.encode([wide, wide_e4012], normalize_embeddings=True)
print(f"\nwide image, right edge replaced: cosine = {float(a @ b):.4f}")
```

**The important lines.**

- `clip[0].model` is the Hugging Face `CLIPModel` inside the Sentence Transformers wrapper, and `clip[0].processor.image_processor` its image processor (steps 1 to 3). `vision_model` runs steps 4 to 8; `visual_projection` is step 9 without the normalisation, which the next lines add.
- `vision_out.last_hidden_state` holds every token's final vector; `pooler_output` is the layer-normalised class token.
- `pixel_vec` is the naive baseline: shrink to 64 by 40 greyscale, centre and normalise, and compare pixel by pixel.
- `.to(hf_model.device)` matters on machines with a GPU or Apple silicon, where the library may have placed the model off the CPU; without it you get a device-mismatch error.

**Output**, from a real run:

```output
pixel tensor:    (1, 3, 224, 224)
token sequence:  (1, 50, 768) (1 class token + 7 x 7 patches)
pooled, 768-d:   (1, 768)
projected:       (1, 512)
same as clip.encode? cosine = 1.0000

variant                    embedding  pixels
half size                      0.991   1.000
JPEG quality 20                0.970   1.000
Gaussian blur                  0.910   1.000
greyscale                      0.968   1.000
shifted 40 px right            0.899   0.946
rotated 10 degrees             0.852   0.478
colours inverted               0.908  -1.000
other error (E-4012)           0.820   0.995
other app, same layout         0.725   0.996
unrelated (floor plan)         0.578   0.014
```

and finally:

```output
wide image, right edge replaced: cosine = 1.0000
```

**What the output means.**

- **The shapes match the nine steps**: a 3 by 224 by 224 tensor, 50 tokens of 768 dimensions, a 768-dimensional pooled vector, a 512-dimensional projection. The manual path and `clip.encode` agree to four decimal places.
- **Read every score against the floor.** The unrelated floor plan scores 0.578. On this model's scale, "nothing in common" is about 0.5 to 0.6, not 0. A score of 0.85 is therefore far less impressive than it sounds.
- **The encoder is strongly invariant to size, compression and colour**: 0.97 to 0.99. Rotation, blur, shifting and inversion cost more, 0.85 to 0.91.
- **Content differences score in the same range as nuisance changes.** A different error code scores 0.820, below a 10-degree rotation of the same screen (0.852) and a blur (0.910). A different application with the same layout scores 0.725. The encoder does tell them apart, but by margins comparable to what a phone photo costs. This is the measured version of the dark-theme failure in the previous lesson.
- **Raw pixels get this exactly backwards.** They call the other error and the other application near-identical (0.995, 0.996) because only a few dark text pixels differ, and call an inverted image the opposite of the original (−1.0). A learned embedding is worth having because it reverses these judgements.
- **The centre crop is total.** Replacing the right quarter of a wide image changes the embedding by nothing at all, because that region never reached the model.

**Common errors.** Calling the processor without `return_tensors="pt"` returns lists that the model rejects. Applying your own resizing and then the processor's resizes twice, with different filters; pass the original image to the processor. Comparing vectors from two different encoders, even two CLIP variants, gives meaningless scores: each model has its own space.

**Extending it.** Run the same table with a self-supervised encoder (for example a DINOv2 model through `transformers`) and compare which rows move. Replace the synthetic screenshots with fifty of your own images and their realistic variants; the table becomes a model-selection test.

## A real-world example

Meridian's Platform team indexed screenshots from the Meridian Reports documentation for visual search, using the image retrieval design from the previous lesson. Report screenshots are wide, typically 1920 by 600 pixels, with a filter panel on the left, the chart in the middle and a legend on the right. Users searching with a photo of the filter panel ("what do these options do?") got the right page only when they happened to photograph the middle of the screen.

The cause was found by running the centre-crop check above on a sample: changing the left third of a report screenshot did not change its embedding at all. Every indexed report had been reduced to its middle square, and every filter panel was invisible to the index.

The team considered three fixes. Letterboxing kept everything but shrank a 1920-pixel image to 224 pixels wide, so the filter labels became unreadable smudges; it helped photos of the whole screen and did little for photos of the panel. A higher-resolution encoder helped somewhat but quadrupled ingestion time. Tiling each screenshot into overlapping squares, about four tiles per image, gave each panel its own vector at usable resolution, and photo queries of any part of the screen found the right tile and therefore the right page. They shipped tiling for wide screenshots only, letterboxing for everything else, and added the crop check to the ingestion test suite so that a future change of model or preprocessor cannot silently reintroduce the problem.

## Common mistakes

- **Choosing an encoder by benchmark rank alone.** The benchmark's notion of similarity may not be yours; a caption-trained model may ignore exactly the detail your users search by.
- **Leaving the default centre crop on wide or tall images.** Content outside the central square never reaches the model, and nothing reports it.
- **Expecting unrelated images to score near zero.** Image embedding spaces are often compressed into a narrow band; thresholds must be set relative to measured scores of unrelated pairs.
- **Preprocessing twice.** Resizing or normalising yourself and then calling the model's processor changes the input distribution and lowers quality.
- **Mixing vectors from different models or versions in one index.** Re-embedding part of a collection with an updated model makes old and new vectors incomparable; re-embed everything or keep separate indexes.
- **Using a visual embedding where the meaning is in printed text.** General encoders compress text-heavy images into layout; route such images through OCR or a page model.

## Best practices

- Write down, before choosing a model, what "similar" must mean for your queries: same category, same look, same object, same text, or same answer.
- Measure invariances on your own images with a table like the one above, and always include an unrelated pair to anchor the scale.
- Check what the image processor does (resize rule, crop, input size) by reading its configuration, and test it with an edge-change check like the one in the code.
- Pad, tile or switch model for images whose aspect ratio is far from square.
- Store the model name, version and preprocessing settings with every vector, so that re-embedding can be planned and mixed vectors detected.
- Keep the original image so that a better encoder can be adopted later without re-ingesting documents.

## Performance considerations

**Compute.** Encoding cost grows with the number of tokens, and attention grows with its square. Moving from B/32 at 224 pixels to B/16 at 224 multiplies attention work per layer by about 15.5, as computed above; to L/14 at 336 by about 133, on a larger model. For a one-off ingestion of tens of thousands of images on a GPU this is usually affordable; for embedding uploaded query images on CPU servers, a small model keeps latency low.

**Tiling multiplies everything.** Four tiles per wide screenshot means four encoder passes at ingestion and four vectors in the index. Applied to all 20,000 images of an assumed collection, that is 80,000 vectors; applied only to the wide fraction, much fewer. Decide per image kind.

**Storage.** A single 512-dimensional float32 vector is 2 KB. Keeping all 49 patch tokens of B/32 at 768 dimensions is $49 \times 768 \times 4 = 150{,}528$ bytes, about 147 KB per image, before any projection or quantisation. Multi-vector representations need a strong reason.

**Batching.** Encoders process images in batches far more efficiently than one at a time; the `encode` call accepts a list and a `batch_size`. Decoding and resizing large images on the CPU can become the bottleneck before the model does, so parallelise preprocessing.

## Summary

- A vision transformer cuts a normalised, fixed-size square into patches, embeds each with its position, adds a class token, runs transformer layers and projects the class token to a unit vector.
- The training objective decides what similar vectors mean: same category (classifiers), describable in similar words (CLIP-style), visually alike in detail (self-supervised) or answering similar questions (page models).
- Preprocessing is part of the model: the default resize and centre crop silently discards the sides of wide images, and small text is lost at 224 pixels.
- Scores live on a model-specific scale; for CLIP ViT-B/32 on these images, unrelated content scores about 0.58, and content changes move scores about as much as rotation or blur.
- Token count, $(S/P)^2 + 1$, drives compute quadratically and multi-vector storage linearly; choose resolution and patch size with that arithmetic.

## Exercises

1. **By hand.** A model uses 448-pixel inputs and 14-pixel patches with 1,024-dimensional tokens. How many tokens does it produce, how many attention pairs per layer, and how many weights does its patch embedding have?

<details><summary>Show solution</summary>

$448 / 14 = 32$, so $32^2 = 1{,}024$ patches plus the class token: 1,025 tokens. Attention pairs: $1{,}025^2 = 1{,}050{,}625$ per layer, 420 times the 2,500 of B/32 at 224. Patch-embedding weights: $3 \times 14^2 \times 1{,}024 = 602{,}112$.

</details>

2. **By hand.** Normalise the pixel (200, 40, 40), the red banner colour, for CLIP. Which channel ends up furthest from zero?

<details><summary>Show solution</summary>

Red: $(200/255 - 0.4815)/0.2686 = (0.7843 - 0.4815)/0.2686 = 1.127$. Green: $(0.1569 - 0.4578)/0.2613 = -1.152$. Blue: $(0.1569 - 0.4082)/0.2758 = -0.911$. Green is furthest from zero, just ahead of red; the banner is as much "very little green" as "a lot of red".

</details>

3. **Code.** Write a `letterbox(img)` function that pads an image to a square with white before encoding, and rerun the wide-image check. What cosine do you get, and what does it tell you?

<details><summary>Show solution</summary>

```python
def letterbox(img, fill="white"):
    side = max(img.size)
    out = Image.new("RGB", (side, side), fill)
    out.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return out

a, b = clip.encode([letterbox(wide), letterbox(wide_e4012)], normalize_embeddings=True)
print(f"letterboxed, right edge replaced: cosine = {float(a @ b):.4f}")
```

```output
letterboxed, right edge replaced: cosine = 0.9388
```

The change now reaches the model and moves the vector, so the right edge is no longer invisible. But 0.939 is above the rotation and blur scores in the invariance table: at 224 pixels for a 1,280-pixel-wide image, the change is small relative to everything else. Letterboxing prevents silent loss; tiling is needed if the edge must be searchable on its own.

</details>

4. **Design.** For each Meridian use case, choose an encoder family and justify it: (a) finding all photos of the same damaged laptop model across support tickets, by uploading one photo; (b) "show me pictures of the new office kitchen" over the facilities photo library; (c) finding the slide in a training deck that answers "how many days can I carry over?".

<details><summary>Show solution</summary>

(a) Self-supervised (DINOv2-style): the query is an image, the similarity wanted is fine visual detail with no words attached, and caption-trained models tend to stop at "a laptop". (b) Language-supervised (CLIP or SigLIP): the query is text about a photograph's general content, exactly what caption training covers; office photos contain little printed text. (c) A page model or OCR text surrogates: the answer is printed text on a slide, which general encoders read poorly. A page model retrieves the slide image directly; OCR text in a normal text index is the cheaper alternative and also allows keyword search.

</details>

## Check your understanding

```quiz
[
  {
    "q": "In CLIP ViT-B/32 at 224 pixels, what does the sequence entering the transformer layers consist of?",
    "options": [
      "224 tokens, one per row of pixels",
      "50 tokens: one class token plus 49 patch embeddings with position embeddings added",
      "512 tokens, one per output dimension",
      "3 tokens, one per colour channel"
    ],
    "answer": 1,
    "why": "224 divided by 32 gives a 7 by 7 grid of patches; each becomes a 768-dimensional token, and a learned class token is prepended."
  },
  {
    "q": "Changing only the right quarter of a 1280 by 400 image leaves its CLIP embedding exactly unchanged. What is the cause?",
    "options": [
      "The default preprocessing resizes the shorter side to 224 and crops the centre square, so the right quarter never reaches the model",
      "CLIP is invariant to all colour changes",
      "The embedding is quantised to too few bits",
      "Cosine similarity ignores vector length"
    ],
    "answer": 0,
    "why": "Resize-then-centre-crop keeps only the middle of a wide image. Letterboxing or tiling keeps the edges."
  },
  {
    "q": "A support engineer wants to find photos of the same laptop with the same visible damage, starting from one photo. Which encoder family is the best first candidate?",
    "options": [
      "A supervised ImageNet classifier",
      "A text embedding model applied to the file name",
      "A page-retrieval model trained on document screenshots",
      "A self-supervised encoder such as DINOv2"
    ],
    "answer": 3,
    "why": "Self-supervised training preserves visual detail that has no name in captions or labels, which suits fine-grained image-to-image search."
  },
  {
    "q": "In the invariance table, the unrelated floor plan scores 0.578 against the VPN screenshot. What is the right conclusion?",
    "options": [
      "The model is broken, because unrelated images should score near 0",
      "Floor plans and dialogs are semantically related",
      "Scores must be read relative to the model's own floor; a threshold of 0.5 would accept unrelated images",
      "Cosine similarity cannot be used for images"
    ],
    "answer": 2,
    "why": "Image embedding spaces often occupy a narrow cone, so unrelated pairs score well above zero. Thresholds come from measured unrelated pairs."
  },
  {
    "q": "Moving from ViT-B/32 to ViT-B/16 at the same 224-pixel input multiplies attention pairs per layer by roughly how much?",
    "options": [
      "2",
      "4",
      "8",
      "15.5"
    ],
    "answer": 3,
    "why": "Tokens go from 50 to 197; attention pairs go from 2,500 to 38,809, about 15.5 times."
  }
]
```

## Further reading

- Dosovitskiy, A. et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (the vision transformer). [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- He, K., Zhang, X., Ren, S. and Sun, J. (2015). *Deep Residual Learning for Image Recognition* (ResNet). [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Chen, T., Kornblith, S., Norouzi, M. and Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR). [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Hugging Face Transformers documentation, the CLIP model page, for the model and image processor classes: [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
