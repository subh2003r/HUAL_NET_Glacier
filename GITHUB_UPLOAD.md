# Upload this repository to GitHub

Do not upload the ZIP file as the repository contents. Unpack the folder first.

From inside `HUAL-Net-Computers-and-Geosciences`:

```bash
git init
git add .
git status
git commit -m "Initial reproducible HUAL-Net release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Before `git add .`, check:

```bash
git status
```

You should NOT see your real `.h5`, `.hdf5`, `.npy`, `.pth` or `.pt` files unless
you intentionally decided to release them.

After pushing, open the GitHub repository in a browser and check:

1. README renders correctly.
2. All source folders are visible.
3. The repository is public.
4. The LICENSE is visible.
5. The README does not contain private machine paths.
6. The code link works without login.

Then use that GitHub repository URL in the manuscript's Code Availability section.
