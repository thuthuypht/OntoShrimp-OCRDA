# Uploading this package to GitHub

1. Extract `OntoShrimp_OCRDA_GitHub_Ready.zip` on your computer.
2. Review `README.md`, especially the data/weights and license notes.
3. Create an empty GitHub repository, for example `OntoShrimp-OCRDA`.
4. Copy the extracted repository contents into the local Git repository.
5. Commit and push.
6. After the public URL is known, update `repository-code` in `CITATION.cff` and the citation section in `README.md`.
7. Before journal submission, create a versioned GitHub Release (for example `v1.0.0`) and consider archiving that release in Zenodo to obtain a DOI.

## Do not add by default

The `.gitignore` excludes common large artifacts such as model checkpoints, dataset ZIPs, raw images, Kaggle outputs, caches, and temporary files. In particular, do not commit the multi-gigabyte 45-model bundle to normal Git history.

## Suggested first commit message

`Release frozen OCRDA-v7.4.4 reproducibility materials and Dataset 3 external results`
