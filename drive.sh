mostRecent=`ls -t ~/data2/behavioralCloning/*.h5 | head -n 1`
echo
echo "Most recent save is $mostRecent."
echo
sleep 2
# Suppress tensorflow warnings
#  (or install tensorflow from source).
#export TF_CPP_MIN_LOG_LEVEL=2
python drive.py "$mostRecent"
