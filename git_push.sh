#!/bin/bash

# Script to automate git add, commit, and push with a fixed commit message "a"

# Add all changes
echo "Adding changes..."
git add .

# Commit with the fixed message "a"
echo "Committing changes with message 'a'..."
git commit -m "a"

# Push to the main branch
echo "Pushing to origin/main..."
git push origin main

echo "Done!"

