mostRecent=`ls -t ~/data2/behavioralCloning/*.h5 | head -n 1`
echo
echo "Most recent snapshot is $mostRecent."
echo
sleep 2
python drive.py "$mostRecent"
