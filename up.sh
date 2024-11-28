mkdir ./config
python ./run-me.py
sleep 5
echo "ZzzzzZZZzzz...."
sleep 10
docker-compose up --build -d
