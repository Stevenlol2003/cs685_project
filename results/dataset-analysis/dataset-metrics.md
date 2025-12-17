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

## Properties That Make the Task Challenging

### 1. Oppositional Structure

- All queries require exactly 2 oppositional claims (pro/con)
- Claims must be directly opposing viewpoints on the same topic

### 2. Multi-Perspective Requirement

- Each claim must have multiple supporting perspectives (avg: 6.0 per query)
- Perspectives must be distinct and non-overlapping

### 3. Evidence Grounding

- Each perspective must be supported by document IDs
- Documents must be retrieved from a corpus of 4,107 evidence documents
- Requires accurate retrieval and relevance matching

### 4. Data Quality Challenges

- 13 duplicate query strings (13 duplicate instances)
- Requires handling of duplicate queries in evaluation

## Example Input/Output Pairs

### Example 1

**Query:** Surrealist Memes: Regression or Progression?

**Claim 1:** Surrealist memes represent progression in meme art.

**Perspectives (pro):**
1. Surreal memes are derived from a varied line of modern art traditions.
2. Surrealist memes represent progression in meme art because they demonstrate that no message is necessary for impactful art.

**Supporting document IDs:** `[205, 364]`...

**Claim 2:** Surreal art is taking meme art back centuries (well, ok, decade).

**Perspectives (con):**
1. Surreal memes are internet elitism at its worst.
2. The perspective that surreal art is taking meme art back centuries is supported by the idea that much of its nuance is "Lost in Translation."

**Supporting document IDs:** `[1138, 858]`...

### Example 2

**Query:** Throwback TV: Friends or Seinfeld?

**Claim 1:** Why Friends is better than Seinfeld

**Perspectives (pro):**
1. Friends is the epitome of a classic sitcom.
2. The show's character development showed depth and heart, making Friends better than Seinfeld.

**Supporting document IDs:** `[199, 482, 1197]`...

**Claim 2:** Why Seinfeld is better than Friends

**Perspectives (con):**
1. Seinfeld is innovative.
2. It redefined the use of characters.

**Supporting document IDs:** `[867, 1094, 783]`...
