REM https://graphite.dev/guides/upstream-remote

git checkout master
git fetch upstream
REM git pull --rebase upstream master
git merge upstream/master
REM git push origin master