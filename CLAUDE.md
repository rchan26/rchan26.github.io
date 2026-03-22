# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What This Is

A static personal portfolio website for Ryan Chan, deployed via GitHub Pages at
`rchan26.github.io`. Custom minimalist design with light and dark modes, no
external CSS frameworks, and minimal JavaScript (theme toggle only).

## Development Commands

**Serve locally (requires Jekyll):**

```bash
jekyll serve
```

**Run pre-commit hooks manually:**

```bash
pre-commit run --all-files
```

**Run a specific hook:**

```bash
pre-commit run prettier --all-files
pre-commit run trailing-whitespace --all-files
```

## Code Quality

Pre-commit hooks enforce:

- **Prettier** (v4.0.0-alpha.8) — formats HTML, CSS, SCSS, JS, Markdown, YAML,
  JSON with `--prose-wrap=always`
- **ShellCheck** — lints shell scripts
- Standard hygiene: trailing whitespace, end-of-file newline, YAML validity, no
  merge conflict markers, no large files

Prettier is the main formatting tool. Run it before committing changes to HTML.

## Architecture

The site is a **single-page application** — all content, CSS, and JS live in
`index.html`. There are no external stylesheets and no Jekyll theme.

- `index.html` — the entire site: inline `<style>` block + sections (header,
  about, research & projects, publications, talks & awards, miscellaneous) +
  inline `<script>` for the theme toggle
- `images/` — profile photo (`profile.jpg`), favicon (`icon.png`)
- `pdfs/` — papers, talks, posters, CV, thesis
- `_config.yml` — minimal Jekyll config (title only; no theme)

## Theming

All colours and typography are driven by CSS custom properties on `:root` and
`[data-theme="dark"]`. The toggle button sets `data-theme` on `<html>` and
persists the choice in `localStorage`.

- **Light mode** — warm cream background (`#fefae0`), Gowun Batang serif (Google
  Fonts), amber-brown accents
- **Dark mode** — solarized dark background (`#002b36`), `Lucida Console`
  monospace at 18px, solarized blue/cyan link colours

To change a colour in both modes, update the corresponding `--variable` in
`:root` (light) and `[data-theme="dark"]` (dark).

## Content Updates

All content changes go in `index.html`. Sections are separated by `<hr />` tags
and labelled with `<h2>` headings. Publications, talks, and project descriptions
are plain HTML list items within those sections.

PDFs (papers, posters, talk slides) go in the appropriate `pdfs/` subdirectory
and are linked directly from `index.html`.
