## Dataset Statistics

- **Number of queries:** 185
- **Number of evidence documents:** 4,107
- **Total words across all documents:** 481,590
- **Average words per document:** 117.3

## Data Quality Metrics

- **Unique queries:** 172
- **Duplicate queries:** 13

**Sample duplicate queries (each appearing 2 times):**
1. "Should we Fight or Embrace Illegal Downloads?"
2. "Disney Animated Films – Are they good for children?"
3. "Should We Cheer for Beauty Pageants or Call for Their Demise?"
4. "Does Nicki Minaj Empower or Objectify women?"
5. "Climate Change – Act now, or question the urgency of the threat?"

## Task-Specific Statistics

- **Average perspectives per query:** 6.0
- **Average words per perspective:** 12.0
- **Average documents per query:** 6.0
- **Average words per claim:** 5.3

## Example Input/Output Pair

**Input:**

Given the query and associated documents, produce a multi-perspective summary that adheres to these standards:
- Generate exactly two oppositional claims (pro/con)
- Each claim must have multiple distinct, non-overlapping perspectives
- Each perspective must be supported by at least one document ID
- Perspectives should be concise, one-sentence summaries

- **query:** "Surrealist Memes: Regression or Progression?"

- **docs:**
  - `"205"`: "Surrealist memes find cultural forbears in the pre-internet art movements of Dadaism, Surrealism and Pop Art. Surrealist memes approach the absurdity of contemporary life in the same manner as Dadaism..."
  - `"364"`: "A central criticism of surreal memes is that they don’t have any point, that they are devoid of any distinct message. To these critics, I (along with troves of surrealist memers) say, so what? Art nee..."
  - `"1138"`: "The internet underground is famously proprietary about its memes. Imageboard website 4chan has long claimed ownership of the internet’s most popular memes, and its users regularly scorn sites like 9ga..."
  - `"858"`: "The nihilism of surreal memes reveals their regressive nature. At its core, art is about evoking emotion and sharing our experiences of the world. Art writer Linda Weintraub writes “If art doesn’t sen..."

Generate the JSON output now.

**Output:**

**Claim 1:** Surrealist memes represent progression in meme art.

**Perspectives (pro):**
1. Surreal memes are derived from a varied line of modern art traditions. (grounded by document `205`)
2. Surrealist memes represent progression in meme art because they demonstrate that no message is necessary for impactful art. (grounded by document `364`)

**Claim 2:** Surreal art is taking meme art back centuries (well, ok, decade).

**Perspectives (con):**
1. Surreal memes are internet elitism at its worst. (grounded by document `1138`)
2. The perspective that surreal art is taking meme art back centuries is supported by the idea that much of its nuance is "Lost in Translation." (grounded by document `858`)
