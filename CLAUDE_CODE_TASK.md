# Claude Code Task: Proofread and Publish tri-processor-inference

## What this is
A new GitHub repo for Peterc3-dev: `tri-processor-inference` — a self-optimizing inference runtime architecture targeting AMD Ryzen AI 300 series APUs (CPU + iGPU + NPU).

## Files in this folder
- `README.md` — Full project README (architecture, feasibility, research refs, build instructions)
- `blog-post.md` — dev.to article with frontmatter (shorter hook piece linking to the repo)

## What I need you to do

### 1. Proofread both files
- Fix any typos, grammar issues, or awkward phrasing
- Verify all URLs are correctly formatted (markdown links)
- Check that technical claims are internally consistent between the two documents
- Ensure the "first of its kind" framing is precise — it says "first open-source architecture" not "first runtime" (because there's no code yet)
- Flag anything that sounds like overclaiming or could damage credibility

### 2. Create the GitHub repo and publish
```bash
cd /path/to/this/folder
git init
gh repo create route-rag-racer --public --description "Route Rag Racer [Adaptive Tri-Processor Inference Runtime] — Self-optimizing CPU+iGPU+NPU inference for AMD Ryzen AI 300 series" --source . --push
```

### 3. After repo is live
- Confirm the repo URL: `https://github.com/Peterc3-dev/route-rag-racer`
- The blog-post.md references this URL — verify the link is correct
- Do NOT publish the blog post to dev.to yet — just confirm it's ready

## Style notes
- Tone is technical but accessible. Not academic, not casual.
- The author is Peter Clemente (@Peterc3-dev)
- This connects to a broader project called CIN (mentioned once at the end of the blog post, not over-explained)
- Tables should render correctly on GitHub

## Do NOT
- Add any code files (this is pre-alpha, architecture only)
- Create an ARCHITECTURE.md yet (referenced in README but not built)
- Change the MIT license choice
- Add CI/CD or GitHub Actions
