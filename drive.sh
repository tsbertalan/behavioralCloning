mostRecent=`ls -t ~/data2/behavioralCloning/*.h5 | head -n 1`
echo "Most recent snapshot is $mostRecent."
python drive.py "$mostRecent"
