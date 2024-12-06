# Machine Learning Reference Values

## Description
This is a knowledge vault used to gather reference implementations of machine learning algorithms that helped me navigate the contents of the Machine Learning Specialization provided by Stanford and DeepLearning.AI.

This project uses **Git Flow** as a branching strategy to organize development. Git Flow helps with version management and parallel development by using dedicated branches for features, fixes, and releases.

## Git Flow Setup
To use Git Flow in this project, install Git Flow. You can install it with:

```bash
# On Debian/Ubuntu-based systems
sudo apt-get install git-flow
```

Then, initialize Git Flow in your repository:

```bash
git flow init
```

This command will prompt you to confirm the naming scheme for the main branches. The names used for the project are:
- **Main branch**: `main`
- **Develop branch**: `dev`
- **Feature branches**: `feature/*`
- **Release branches**: `release/*`
- **Hotfix branches**: `hotfix/*`
- **Support branches**: `support/*`

## Workflow

### 1. Main branches
- **main**: contains production code.
- **develop**: contains the development code, integrated from different features.

### 2. Main Commands
For features, releases, hotfixes, and support, the branch name should start with `ML`, which are the initials of the Machine Learning class.

#### Create a new feature
To develop a new feature, create a `feature` branch from `develop`:

```bash
git flow feature start feature-name
```

Once the feature is complete, finish the branch:

```bash
git flow feature finish feature-name
```

#### Prepare a release version
To prepare a new release, create a `release` branch from `develop`:

```bash
git flow release start version-number
```

When the release is ready, finish the branch:

```bash
git flow release finish version-number
```

This will:
1. Merge the `release` branch into `main` and `develop`.
2. Create a tag in `main` with the version number.

#### Create an emergency fix (Hotfix)
To make a quick fix in `main`, create a `hotfix` branch from `main`:

```bash
git flow hotfix start hotfix-name
```

When the fix is complete, finish the branch:

```bash
git flow hotfix finish hotfix-name
```

This will:
1. Merge the `hotfix` branch into `main` and `develop`.
2. Create a tag in `main` with the updated version number.

#### Create a support branch (Support)
Support branches are used to maintain old versions that require maintenance or support while continuing development on the main branch. To create a support branch, use the following command:

```bash
git flow support start support-version-name
```

This branch is created from `main` and is used to make specific changes to a version that is already in production, without affecting active development in `develop`.

### Explanation
This `README.md` includes:
1. **Installation and configuration** of Git Flow.
2. **Key commands** and their purpose in the workflow.
3. **Description of main branches** and how to start and finish each type of branch (`feature`, `release`, `hotfix`, and `support`).
