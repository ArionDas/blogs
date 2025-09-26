# ML Research Blog

A modern, fast, and feature-rich blog built with Hugo and the PaperMod theme. Perfect for machine learning research, technical content, and academic writing.

## Features

### **Performance & Technology**
- **Hugo Static Site Generator**: Lightning-fast site generation and loading
- **PaperMod Theme**: Clean, modern, and responsive design
- **GitHub Pages Deployment**: Automated deployment with GitHub Actions
- **Version Control**: All content versioned with Git

### **Content Features**
- **Markdown Writing**: Write posts in clean, readable Markdown
- **Code Highlighting**: Syntax highlighting for 100+ programming languages
- **LaTeX Math Support**: Full mathematical expression rendering with MathJax
- **Image Support**: High-quality images with optimization
- **Table of Contents**: Auto-generated TOC for long posts
- **Reading Time**: Estimated reading time for each post
- **Search Functionality**: Fast client-side search

### **Interactive Elements**
- **Like/Dislike Buttons**: Powered by Lyket for reader engagement
- **Social Sharing**: Share posts across platforms
- **Comments System**: Ready for integration with various comment systems
- **Dark/Light Mode**: Theme toggle for better reading experience

### **Developer Experience**
- **Live Reload**: Local development server with hot reload
- **SEO Optimized**: Built-in SEO features and meta tags
- **Mobile Responsive**: Looks great on all devices
- **Fast Builds**: Optimized build process

## Quick Start

### Prerequisites
- [Hugo Extended](https://gohugo.io/installation/) (v0.135.0 or later)
- [Git](https://git-scm.com/)
- [Node.js](https://nodejs.org/) (optional, for advanced features)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ArionDas/blogs.git
   cd blogs
   ```

2. **Initialize and update submodules**:
   ```bash
   git submodule update --init --recursive
   ```

3. **Start the development server**:
   ```bash
   hugo server -D
   ```

4. **Open your browser** and navigate to `http://localhost:1313`

## Creating Content

### Create a New Post

```bash
hugo new content posts/my-new-post.md
```

### Post Structure

Each post should have frontmatter like this:

```yaml
---
title: "Your Post Title"
date: 2025-09-25T21:57:40Z
draft: false
tags: ["machine learning", "python", "tutorial"]
categories: ["Deep Learning"]
author: "Your Name"
description: "Brief description of your post"
showToc: true
math: true
---

Your content here...
```

### Writing Math

Use LaTeX syntax for mathematical expressions:

- Inline math: `$E = mc^2$`
- Display math: `$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$$`

### Code Blocks

Use fenced code blocks with language specification:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)
```

### Adding Images

1. Place images in `static/images/`
2. Reference them in posts:
   ```markdown
   ![Alt text](/images/my-image.jpg)
   ```

### Like/Dislike Buttons

Add interactive buttons to your posts:

```html
<div id="lyket-like-my-post-id"></div>
```

## Customization

### Site Configuration

Edit `hugo.yaml` to customize:
- Site title and description
- Author information
- Social media links
- Theme settings
- Analytics

### Theme Customization

- **Colors**: Modify CSS variables in `assets/css/extended/`
- **Layouts**: Override layouts in `layouts/`
- **Partials**: Customize partials in `layouts/partials/`

### Adding Custom CSS

Create `assets/css/extended/custom.css`:

```css
:root {
    --primary-color: #your-color;
}

.custom-class {
    /* your styles */
}
```

## Deployment

### GitHub Pages (Recommended)

1. **Enable GitHub Pages** in your repository settings
2. **Set source** to "GitHub Actions"
3. **Push to main branch** - deployment happens automatically

The included GitHub Actions workflow (`.github/workflows/hugo.yml`) handles:
- Hugo installation
- Site building
- Deployment to GitHub Pages

### Manual Deployment

```bash
# Build the site
hugo --gc --minify

# Deploy the public/ directory to your hosting provider
```

### Other Hosting Options

- **Netlify**: Connect your GitHub repo for automatic deployments
- **Vercel**: Import project and deploy
- **Firebase Hosting**: Use Firebase CLI
- **AWS S3**: Sync public/ directory to S3 bucket

## Development

### Project Structure

```
blogs/
├── .github/workflows/     # GitHub Actions
├── archetypes/           # Content templates
├── assets/              # CSS, JS, images
├── content/             # Blog posts and pages
│   └── posts/          # Blog posts
├── layouts/             # Custom layouts and partials
│   └── partials/       # Reusable components
├── static/              # Static files
│   └── images/         # Post images
├── themes/              # Hugo themes
│   └── PaperMod/       # PaperMod theme (submodule)
├── hugo.yaml           # Site configuration
└── README.md
```

### Useful Commands

```bash
# Create new post
hugo new content posts/post-name.md

# Start development server
hugo server -D

# Build site
hugo --gc --minify

# Update theme
git submodule update --remote themes/PaperMod
```

### Local Development Tips

1. **Draft Posts**: Set `draft: true` in frontmatter, use `-D` flag to preview
2. **Live Reload**: Hugo automatically refreshes browser on changes
3. **Port Configuration**: Use `hugo server --port 8080` for custom port
4. **Bind Address**: Use `hugo server --bind 0.0.0.0` to access from other devices

## Configuration

### Environment Variables

For production deployment, you may need:

```bash
HUGO_ENV=production
HUGO_ENVIRONMENT=production
```

### Analytics Setup

Edit `hugo.yaml` to add analytics:

```yaml
params:
  analytics:
    google:
      SiteVerificationTag: "your-verification-tag"
```

### Lyket Integration

1. Sign up at [Lyket.dev](https://lyket.dev/)
2. Get your API key
3. Replace the placeholder in `layouts/partials/extend_footer.html`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [Hugo Documentation](https://gohugo.io/documentation/)
- **Theme**: [PaperMod Documentation](https://github.com/adityatelange/hugo-PaperMod)
- **Issues**: Create an issue in this repository

## Acknowledgments

- [Hugo](https://gohugo.io/) - The world's fastest framework for building websites
- [PaperMod](https://github.com/adityatelange/hugo-PaperMod) - A fast, clean, responsive Hugo theme
- [MathJax](https://www.mathjax.org/) - Beautiful math in all browsers
- [Lyket](https://lyket.dev/) - Like buttons for websites

---

**Happy blogging!**

Perfect for ML model blogs with math rendering, code highlighting, fast deployment, and interactive features!