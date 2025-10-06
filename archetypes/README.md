# Blog Post Templates (Archetypes)

This directory contains Hugo archetypes - templates for creating new blog posts with pre-filled structure and placeholders.

## Available Templates

### 1. Model Family Template (`model-family.md`)

**Purpose**: Writing about ML models and architectures

**Best for**:
- Neural network architectures (GPT, BERT, ResNet, etc.)
- Model families (Transformers, CNNs, RNNs, etc.)
- Novel architectures from research papers

**Usage**:
```bash
hugo new content posts/gpt-architecture.md --kind model-family
```

**Includes**:
- Architecture overview and diagrams
- Mathematical foundations and formulas
- PyTorch implementation examples
- Training details and hyperparameters
- Benchmark results and comparisons
- Applications and use cases

---

### 2. System Design Template (`system-design.md`)

**Purpose**: Writing about system architecture and design patterns

**Best for**:
- Distributed systems concepts
- Scalability patterns
- System architecture designs
- Engineering best practices

**Usage**:
```bash
hugo new content posts/caching-strategies.md --kind system-design
```

**Includes**:
- Requirements analysis
- Architecture diagrams
- Design decisions and trade-offs
- Scalability considerations
- Implementation details
- Real-world examples

---

### 3. Topic Knowledge Template (`topic-knowledge.md`)

**Purpose**: Educational content and concept explanations

**Best for**:
- Reinforcement Learning concepts
- Attention mechanisms
- NLP fundamentals
- Optimization algorithms
- General ML/AI topics
- Tutorial-style content

**Usage**:
```bash
hugo new content posts/reinforcement-learning.md --kind topic-knowledge
```

**Includes**:
- Prerequisites and background
- Fundamental concepts breakdown
- Mathematical foundations
- Step-by-step examples
- Implementation code
- Applications and best practices

---

### 4. Default Template (`default.md`)

**Purpose**: Simple, minimal template for general posts

**Usage**:
```bash
hugo new content posts/my-post.md
# OR without --kind flag (uses default)
```

## Customizing Templates

All templates are fully configurable. To customize:

1. **Edit the template file** in `archetypes/` directory
2. **Modify sections** - Add, remove, or reorder sections
3. **Update frontmatter** - Change default tags, categories, or metadata
4. **Add/remove placeholders** - Adjust placeholder text and examples

### Frontmatter Variables

Hugo templates support Go template syntax with these variables:

- `{{ .File.ContentBaseName }}` - The filename without extension
- `{{ .Date }}` - Current date/time
- `{{ replace .File.ContentBaseName "-" " " | title }}` - Filename as title

### Example Customization

To add a new section to the model-family template:

1. Open `archetypes/model-family.md`
2. Add your section in the appropriate location:
   ```markdown
   ## My Custom Section
   
   <!-- Custom content here -->
   ```
3. Save the file
4. New posts created with `--kind model-family` will include your section

## Tips for Using Templates

1. **Choose the right template** - Pick the one that best matches your content type
2. **Don't skip placeholders** - Fill in all `[placeholder]` text
3. **Remove unused sections** - Delete sections that aren't relevant
4. **Keep structure consistent** - Maintain the organizational structure for easier navigation
5. **Update frontmatter** - Always customize title, tags, categories, and description
6. **Set draft status** - Keep `draft: true` until ready to publish

## Template Structure Philosophy

All templates follow these principles:

1. **Comprehensive coverage** - Include all possible sections that might be needed
2. **Easy to remove** - Unused sections can be easily deleted
3. **Clear placeholders** - All customizable content is clearly marked
4. **Consistent formatting** - Match the style of existing blog posts
5. **Best practices** - Include examples of good structure and content organization

## Examples

### Creating a Post About GPT-4

```bash
hugo new content posts/gpt4-architecture.md --kind model-family
```

Then customize:
- Replace `[Model Name]` with "GPT-4"
- Fill in architecture details
- Add implementation examples
- Include benchmark results

### Creating a Post About Load Balancing

```bash
hugo new content posts/load-balancing-strategies.md --kind system-design
```

Then customize:
- Replace `[System/Concept Name]` with "Load Balancing"
- Fill in architecture patterns
- Add scalability considerations
- Include real-world examples

### Creating a Tutorial on Attention Mechanisms

```bash
hugo new content posts/attention-mechanisms.md --kind topic-knowledge
```

Then customize:
- Replace `[Topic Name]` with "Attention Mechanisms"
- Fill in fundamental concepts
- Add mathematical derivations
- Include code examples

## Need Help?

- Check the [Hugo Documentation](https://gohugo.io/content-management/archetypes/)
- Look at existing posts in `content/posts/` for examples
- Refer to `content/posts/transformer-architecture.md` as a complete example

## Contributing

To add a new template:

1. Create a new `.md` file in this directory
2. Follow the existing template structure
3. Add documentation to this README
4. Update the main README.md with usage instructions
