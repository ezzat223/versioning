# Commands

```bash
## Manage and explore repos
lakectl repo
    create      # Create a new repository
    delete      # Delete existing repository
    list        # List repositories

## Commit changes on a given branch
lakectl commit

## Create and manage branches within a repository
lakectl branch
    create      # Create a new branch in a repository
    delete      # Delete a branch in a repository, along with its uncommitted changes (CAREFUL)
    list        # List branches in a repository
    reset       # Reset uncommitted changes - all of them, or by path
    revert      # Given a commit, record a new commit to reverse the effect of this commit
    show        # Show branch latest commit reference

## Sync local directories with lakeFS paths
lakectl local
    checkout <local-dir>   # Sync local directory with the remote state.
    clone lakefs://<repo-name>/<branch>/<dir-name> <local-dir>       # Clones lakeFS data from a path into an empty local directory and initializes the directory - A directory can only track a single lakeFS remote location. i.e., you cannot clone data into an already initialized directory
        --pre-sign=false # You might face an issue, so use this bypasses the presigned URL issue entirely.
    commit      # Commit changes from local directory to the lakeFS branch it tracks.
    init        # Connects between a local directory and a lakeFS remote URI to enable data sync - To undo a directory init, delete the .lakefs_ref.yaml file created in the initialized directory
    list        # find and list local directories that are synced with lakeFS - It is recommended to follow any init or clone command with a list command to verify its success
    pull        # Fetches latest changes from a lakeFS remote location into a connected local directory
    status      # show modifications (both remote and local) to the directory and the remote location it tracks - Verify that your local environment is up-to-date with its remote path, if not, then use lakectl local checkout

## Merge & commit changes from source branch into destination branch
lakectl merge

## View and manipulate objects
lakectl fs
    cat             # Dump content of object to stdout
    download        # Download object(s) from a given repository path
        --pre-sign=false # You might face an issue, so use this bypasses the presigned URL issue entirely.
    ls              # List entries under a given tree
    presign         # return a pre-signed URL for reading the specified object
    rm              # Delete object
    stat            # View object metadata
    upload          # Upload a local file to the specified URI
        --pre-sign=false # You might face an issue, so use this bypasses the presigned URL issue entirely.

## Import data from external source to a destination branch
lakectl import --from <object store URI> --to <lakeFS path URI> [flags]

## Create and manage branch protection rules
lakectl branch-protect

## Create and manage tags within a repository
lakectl tag

## Apply the changes introduced by an existing commit
lakectl cherry-pick

## Manage Actions commands
lakectl actions
    runs            # Explore runs information
        describe        # Describe run results
        list            # List runs
    validate        # Validate action file

## List entries under a given path, annotating each with the latest modifying commit
lakectl annotate

---

## Show changes between two commits, or the currently uncommitted changes
lakectl diff

## Show log of commits for a given reference
lakectl log lakefs://<repo-name>/<branch>

## See detailed information about a commit
lakectl show commit <commit-URI>

---

## Use a web browser to log into lakeFS
lakectl login

## Manage authentication and authorization
lakectl auth

## Show the info of the configured user in lakectl
lakectl identity

## Manage lakectl plugins
lakectl plugin

## Manage the garbage collection policy
lakectl gc
    delete-config   # Deletes the garbage collection policy for the repository
    get-config      # Show the garbage collection policy for this repository
    set-config      # Set garbage collection policy JSON

## Create/update local lakeFS configuration
lakectl config

## Run a basic diagnosis of the LakeFS configuration
lakectl doctor

## Generate completion script
lakectl completion
```
