# Practical Exam Execution Guide (Step-by-Step)

This guide is structured for direct exam execution. For each practical:
1. **Files to Create** (exact file names, locations, and complete working code with all fixes applied)
2. **Install Dependencies** (exact terminal command)
3. **Run Commands** (exact step-by-step commands to build and execute)
4. **Expected Output** (what to show the examiner)
5. **Cleanup / Teardown** (commands to stop services)

---

# Table of Contents
- [Practical 1: Spark Data Skew Fix (Key Salting)](#practical-1-spark-data-skew-fix-key-salting)
- [Practical 2: Delta Lake ACID Transactions & Time Travel](#practical-2-delta-lake-acid-transactions--time-travel)
- [Practical 3: 3-Broker Kafka Cluster in KRaft Mode](#practical-3-3-broker-kafka-cluster-in-kraft-mode)
- [Practical 4: Avro Schema Evolution & Compatibility (Schema Registry)](#practical-4-avro-schema-evolution--compatibility-schema-registry)
- [Practical 5: Spark Structured Streaming Word Count (TCP Socket)](#practical-5-spark-structured-streaming-word-count-tcp-socket)
- [Practical 6: Stateful Fraud Detection with Apache Flink (Java)](#practical-6-stateful-fraud-detection-with-apache-flink-java)
- [Practical 7: Watermarks & Late Data Handling with Apache Flink (PyFlink)](#practical-7-watermarks--late-data-handling-with-apache-flink-pyflink)
- [Practical 8: Graph Processing & PageRank using Spark GraphX (Scala / SBT)](#practical-8-graph-processing--pagerank-using-spark-graphx-scala--sbt)
- [Practical 9: Federated SQL Queries with Trino & MySQL](#practical-9-federated-sql-queries-with-trino--mysql)
- [Practical 10: End-to-End Real-Time Analytics (Kafka -> ClickHouse -> Grafana)](#practical-10-end-to-end-real-time-analytics-kafka---clickhouse---grafana)

---

## Practical 1: Spark Data Skew Fix (Key Salting)

### 1. Files to Create
Create directory `P1` and file `P1/skew.py`:

```python
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, rand, concat_ws, sum as _sum
)

# 1. Initialize Spark Session
spark = (
    SparkSession.builder
    .appName("SparkSkewPractical")
    .master("local[4]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("Spark application started!")

# 2. Create Skewed Dataset (90% records have key = 0)
number_of_records = 5_000_000
df = spark.range(number_of_records)

skewed_df = df.withColumn(
    "key",
    when(
        col("id") < number_of_records * 0.90,
        lit(0)
    ).otherwise(
        (col("id") % 999) + 1
    )
)

print("Skewed dataset created!")
print("\nKey distribution (Top 10):")
skewed_df.groupBy("key").count().orderBy(
    col("count").desc()
).show(10)

# 3. Apply Salting Fix (Add random salt 0..7)
print("\nStarting SALTED GROUP BY (Data Skew Fix)...")
salted_df = skewed_df.withColumn(
    "salt", (rand() * 8).cast("int")
)
salted_df = salted_df.withColumn(
    "salted_key",
    concat_ws("_", col("key"), col("salt"))
)

# 4. Two-Stage Aggregation
partial_count = salted_df.groupBy(
    "salted_key", "key"
).count()

final_result = (
    partial_count
    .groupBy("key")
    .agg(_sum("count").alias("count"))
    .orderBy(col("count").desc())
)

# 5. Display Salted Aggregation Result
final_result.show(10)

print("\nSALTED GROUP BY completed successfully!")
print("Open Spark UI: http://localhost:4040")

time.sleep(120)
spark.stop()
```

### 2. Install Dependencies
```powershell
pip install pyspark
```
*(Ensure Java JDK 17 / JDK 11 is installed and `JAVA_HOME` is set. On Windows, ensure `C:\hadoop\bin\winutils.exe` is present).*

### 3. Run Commands
```powershell
cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P1"
python skew.py
```

### 4. Expected Output
- Key distribution showing heavy skew on key `0` (~4,500,000 rows).
- Salted group by completes without single-task worker bottleneck.
- Spark UI viewable at `http://localhost:4040` under the **Stages** tab.

---

## Practical 2: Delta Lake ACID Transactions & Time Travel

### 1. Files to Create
Create directory `P2` and file `P2/dl.py`:

```python
import os
import sys

# Configure Hadoop and Python environments for Windows
os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip, DeltaTable

# 1. Initialize Spark Session with Delta Extensions
builder = (
    SparkSession.builder
    .appName("DeltaLakeDemo")
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("Spark started successfully!")

# 2. Create Initial Dataset
data = [
    (1, "Siddhant", 90),
    (2, "Shreya", 88),
    (3, "Shruti", 75),
    (4, "Chinmay", 90),
    (5, "Vishnu", 88),
    (6, "Nikita", 75)
]

df = spark.createDataFrame(data, ["id", "name", "marks"])
print("Student Data:")
df.show()

# 3. Write Data in Delta Format
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delta-students")
df.write.format("delta").mode("overwrite").save(path)

print("Read back from Delta table:")
spark.read.format("delta").load(path).show()

# 4. Perform ACID Update
dt = DeltaTable.forPath(spark, path)
dt.update(
    condition="name = 'Nikita'",
    set={"marks": "80"}
)

print("After update:")
spark.read.format("delta").load(path).show()

# 5. Time Travel (Read Version 0)
print("Version 0 (before update):")
spark.read.format("delta").option("versionAsOf", 0).load(path).show()

# 6. Show Transaction History
print("Table history:")
dt.history().select("version", "timestamp", "operation").show(truncate=False)

spark.stop()
```

### 2. Install Dependencies
```powershell
pip install pyspark delta-spark
```

### 3. Run Commands
```powershell
cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P2"
python dl.py
```

### 4. Expected Output
- Initial table of 6 students.
- Updated table where Nikita's marks change to `80`.
- Version 0 table where Nikita's marks are still `75`.
- Audit history showing version 0 (`WRITE`) and version 1 (`UPDATE`).

---

## Practical 3: 3-Broker Kafka Cluster in KRaft Mode

### 1. Files to Create
Create directory `P3` and the following 3 files:

#### File: `P3/docker-compose.yml`
```yaml
services:
  kafka1:
    image: bitnamilegacy/kafka:latest
    container_name: kafka1
    ports:
      - "29092:29092"
    environment:
      - KAFKA_KRAFT_CLUSTER_ID=abcdefghijklmnopqrstuv
      - KAFKA_CFG_NODE_ID=1
      - KAFKA_CFG_PROCESS_ROLES=broker,controller
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=1@kafka1:9093,2@kafka2:9093,3@kafka3:9093
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093,EXTERNAL://:29092
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://kafka1:9092,EXTERNAL://localhost:29092
      - KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT,EXTERNAL:PLAINTEXT
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CFG_INTER_BROKER_LISTENER_NAME=PLAINTEXT
      - KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE=true
      - ALLOW_PLAINTEXT_LISTENER=yes
    volumes:
      - kafka1_data:/bitnami/kafka

  kafka2:
    image: bitnamilegacy/kafka:latest
    container_name: kafka2
    ports:
      - "39092:29092"
    environment:
      - KAFKA_KRAFT_CLUSTER_ID=abcdefghijklmnopqrstuv
      - KAFKA_CFG_NODE_ID=2
      - KAFKA_CFG_PROCESS_ROLES=broker,controller
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=1@kafka1:9093,2@kafka2:9093,3@kafka3:9093
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093,EXTERNAL://:29092
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://kafka2:9092,EXTERNAL://localhost:39092
      - KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT,EXTERNAL:PLAINTEXT
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CFG_INTER_BROKER_LISTENER_NAME=PLAINTEXT
      - KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE=true
      - ALLOW_PLAINTEXT_LISTENER=yes
    volumes:
      - kafka2_data:/bitnami/kafka

  kafka3:
    image: bitnamilegacy/kafka:latest
    container_name: kafka3
    ports:
      - "49092:29092"
    environment:
      - KAFKA_KRAFT_CLUSTER_ID=abcdefghijklmnopqrstuv
      - KAFKA_CFG_NODE_ID=3
      - KAFKA_CFG_PROCESS_ROLES=broker,controller
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=1@kafka1:9093,2@kafka2:9093,3@kafka3:9093
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093,EXTERNAL://:29092
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://kafka3:9092,EXTERNAL://localhost:49092
      - KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT,EXTERNAL:PLAINTEXT
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CFG_INTER_BROKER_LISTENER_NAME=PLAINTEXT
      - KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE=true
      - ALLOW_PLAINTEXT_LISTENER=yes
    volumes:
      - kafka3_data:/bitnami/kafka

volumes:
  kafka1_data:
  kafka2_data:
  kafka3_data:
```

#### File: `P3/consumer.py`
```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "student-events",
    bootstrap_servers=[
        "localhost:29092",
        "localhost:39092",
        "localhost:49092"
    ],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="student-consumer-group",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("Waiting for messages...\n")

for message in consumer:
    print("Consumed:", message.value, "Partition:", message.partition, "Offset:", message.offset)
```

#### File: `P3/producer.py`
```python
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers=[
        "localhost:29092",
        "localhost:39092",
        "localhost:49092"
    ],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

for i in range(10):
    message = {
        "student_id": i + 1,
        "name": f"Student-{i+1}",
        "course": "CS",
        "message": f"Hello from student {i+1}"
    }
    producer.send("student-events", value=message)
    print("Produced:", message)
    time.sleep(1)

producer.flush()
producer.close()
```

### 2. Install Dependencies
```powershell
pip install kafka-python
```
*(Ensure Docker Desktop is running).*

### 3. Run Commands
1. **Start the Kafka Cluster**:
   ```powershell
   cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P3"
   docker compose up -d
   ```
2. **(Optional) Create Replicated Topic**:
   ```powershell
   docker exec kafka1 /opt/bitnami/kafka/bin/kafka-topics.sh --bootstrap-server kafka1:9092 --create --if-not-exists --topic student-events --partitions 3 --replication-factor 3
   ```
3. **Open Terminal 1 (Run Consumer)**:
   ```powershell
   cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P3"
   python consumer.py
   ```
4. **Open Terminal 2 (Run Producer)**:
   ```powershell
   cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P3"
   python producer.py
   ```

### 4. Expected Output
- Producer prints 10 messages produced to the topic.
- Consumer prints each message along with its assigned partition (0, 1, or 2) and offset.

### 5. Cleanup
```powershell
docker compose down
```

---

## Practical 4: Avro Schema Evolution & Compatibility (Schema Registry)

### 1. Files to Create
Create directory `P4`, subfolder `P4/schemas`, and the following 4 files:

#### File: `P4/docker-compose.yml`
```yaml
services:
  kafka:
    image: confluentinc/cp-kafka:7.6.0
    hostname: kafka
    container_name: kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://kafka:29092,CONTROLLER://kafka:29093,PLAINTEXT_HOST://0.0.0.0:9092
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:29093
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      CLUSTER_ID: MkU3OEVBNTcwNTJENDM2Qk

  schema-registry:
    image: confluentinc/cp-schema-registry:7.6.0
    hostname: schema-registry
    container_name: schema-registry
    depends_on:
      - kafka
    ports:
      - "8081:8081"
    environment:
      SCHEMA_REGISTRY_HOST_NAME: schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka:29092
      SCHEMA_REGISTRY_LISTENERS: http://0.0.0.0:8081
```

#### File: `P4/schemas/patient-v1.avsc`
```json
{
  "type": "record",
  "name": "Patient",
  "namespace": "com.hospital",
  "fields": [
    {"name": "patient_id", "type": "int"},
    {"name": "patient_name", "type": "string"},
    {"name": "age", "type": "int"},
    {"name": "heart_rate", "type": "int"}
  ]
}
```

#### File: `P4/schemas/patient-v2.avsc`
```json
{
  "type": "record",
  "name": "Patient",
  "namespace": "com.hospital",
  "fields": [
    {"name": "patient_id", "type": "int"},
    {"name": "patient_name", "type": "string"},
    {"name": "age", "type": "int"},
    {"name": "heart_rate", "type": "int"},
    {"name": "blood_pressure", "type": "string", "default": "Unknown"}
  ]
}
```

#### File: `P4/pract4.py`
```python
import json
import time
import urllib.request
import urllib.error

REG = "http://localhost:8081"
SUBJECT = "patient-data-value"
CT = {"Content-Type": "application/vnd.schemaregistry.v1+json"}

def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        REG + path, data=data, method=method, headers=CT
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or "null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or "null")

def schema_body(path):
    with open(path) as f:
        return {"schema": json.dumps(json.load(f))}

def step(n, title):
    print(f"\n{'='*70}\nSTEP {n}: {title}\n{'='*70}")

# STEP 1: Wait for Schema Registry readiness
step(1, "Wait for Schema Registry")
for _ in range(30):
    try:
        print("Registry is up. Existing subjects:", call("GET", "/subjects")[1])
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("Schema Registry did not become available")

call("DELETE", f"/subjects/{SUBJECT}")
call("DELETE", f"/subjects/{SUBJECT}?permanent=true")

# STEP 2: Register V1 Schema
step(2, "Register V1 (patient_id, patient_name, age, heart_rate)")
print(call("POST", f"/subjects/{SUBJECT}/versions", schema_body("schemas/patient-v1.avsc")))

# STEP 3: Set Subject Compatibility to BACKWARD
step(3, "Set compatibility of subject to BACKWARD")
print(call("PUT", f"/config/{SUBJECT}", {"compatibility": "BACKWARD"}))
print("Current config:", call("GET", f"/config/{SUBJECT}")[1])

# STEP 4: Test V2 for Compatibility
step(4, "Check V2 (adds blood_pressure WITH default) for compatibility")
print(call("POST", f"/compatibility/subjects/{SUBJECT}/versions/latest", schema_body("schemas/patient-v2.avsc")))

# STEP 5: Register V2 Schema
step(5, "Register V2")
print(call("POST", f"/subjects/{SUBJECT}/versions", schema_body("schemas/patient-v2.avsc")))
print("-> V2 ACCEPTED")

# STEP 6: Test Incompatible V3 Schema
step(6, "Check incompatible V3 (age int->string, diagnosis without default)")
v3 = {
    "type": "record",
    "name": "Patient",
    "namespace": "com.hospital",
    "fields": [
        {"name": "patient_id", "type": "int"},
        {"name": "patient_name", "type": "string"},
        {"name": "age", "type": "string"},
        {"name": "heart_rate", "type": "int"},
        {"name": "blood_pressure", "type": "string", "default": "Unknown"},
        {"name": "diagnosis", "type": "string"}
    ]
}
print(call("POST", f"/compatibility/subjects/{SUBJECT}/versions/latest", {"schema": json.dumps(v3)}))
```

### 2. Install Dependencies
*(No special Python pip dependencies needed, uses standard library `urllib` and `json`).*

### 3. Run Commands
```powershell
cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P4"
docker compose up -d
python pract4.py
```

### 4. Expected Output
- Step 2: `(200, {'id': 1})` (V1 registered).
- Step 4: `(200, {'is_compatible': True})`.
- Step 5: `(200, {'id': 2})` (V2 registered).
- Step 6: `(200, {'is_compatible': False})` (V3 rejected due to breaking BACKWARD compatibility).

### 5. Cleanup
```powershell
docker compose down
```

---

## Practical 5: Spark Structured Streaming Word Count (TCP Socket)

### 1. Files to Create
Create directory `P5` and the following 2 files:

#### File: `P5/socket_sender.py`
```python
import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 9999))
server.listen(1)
print("Waiting for Spark to connect on port 9999...")

conn, _ = server.accept()
print("Spark connected! Type words and press Enter (type 'exit' to stop):")

try:
    while True:
        msg = input("> ")
        if msg.lower() == "exit":
            break
        conn.sendall((msg + "\n").encode("utf-8"))
finally:
    conn.close()
    server.close()
```

#### File: `P5/word_count.py`
```python
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split

os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

spark = SparkSession.builder.master("local[2]").appName("LiveWordCount").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

lines = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()
words = lines.select(explode(split(lines.value, " ")).alias("word"))
word_counts = words.groupBy("word").count()

query = word_counts.writeStream.outputMode("complete").format("console").start()
query.awaitTermination()
```

### 2. Install Dependencies
```powershell
pip install pyspark
```

### 3. Run Commands
1. **Start Terminal 1 FIRST (Socket Server)**:
   ```powershell
   cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P5"
   python socket_sender.py
   ```
2. **Start Terminal 2 (Spark Streaming Job)**:
   ```powershell
   cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P5"
   python word_count.py
   ```
3. In Terminal 1, type text (e.g., `spark streaming hello spark`) and press Enter.

### 4. Expected Output
Terminal 2 prints updated word count tables after each batch:
```
+---------+-----+
|     word|count|
+---------+-----+
|    hello|    1|
|    spark|    2|
|streaming|    1|
+---------+-----+
```

---

## Practical 6: Stateful Fraud Detection with Apache Flink (Java)

### 1. Files to Create
Create directory `P6`, directory `P6/src/main/java`, and the following 2 files:

#### File: `P6/pom.xml`
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">

    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>flink-practical</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <maven.compiler.release>17</maven.compiler.release>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <flink.version>1.19.2</flink.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.apache.flink</groupId>
            <artifactId>flink-streaming-java</artifactId>
            <version>${flink.version}</version>
        </dependency>
        <dependency>
            <groupId>org.apache.flink</groupId>
            <artifactId>flink-clients</artifactId>
            <version>${flink.version}</version>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.13.0</version>
            </plugin>
        </plugins>
    </build>
</project>
```

#### File: `P6/src/main/java/FraudDetectionJob.java`
```java
import org.apache.flink.api.common.state.*;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.streaming.api.environment.*;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import java.io.Serializable;

public class FraudDetectionJob {

    public static class Transaction implements Serializable {
        public String customerId;
        public double amount;
        public long timestamp;

        public Transaction() {}

        public Transaction(String id, double amt, long time) {
            customerId = id;
            amount = amt;
            timestamp = time;
        }
    }

    public static class Detector extends KeyedProcessFunction<String, Transaction, String> {
        private transient ValueState<Transaction> previous;

        public void open(org.apache.flink.configuration.Configuration c) {
            previous = getRuntimeContext().getState(
                new ValueStateDescriptor<>("previous", Types.POJO(Transaction.class)));
        }

        public void processElement(Transaction tx, Context ctx, Collector<String> out) throws Exception {
            if (tx.amount < 10000) return;

            Transaction p = previous.value();
            if (p != null) {
                long gap = tx.timestamp - p.timestamp;
                if (gap >= 0 && gap <= 60000) {
                    out.collect("FRAUD ALERT: " + tx.customerId +
                        ", Previous=₹" + p.amount +
                        ", Current=₹" + tx.amount +
                        ", Gap=" + gap / 1000 + " seconds");
                }
            }
            previous.update(tx);
        }
    }

    public static void main(String[] args) throws Exception {
        var env = StreamExecutionEnvironment.getExecutionEnvironment();
        long base = 1792404000000L;

        env.fromData(
            new Transaction("C101", 15000, base + 10000),
            new Transaction("C202", 5000, base + 20000),
            new Transaction("C101", 20000, base + 45000),
            new Transaction("C202", 12000, base + 50000),
            new Transaction("C101", 3000, base + 55000),
            new Transaction("C101", 18000, base + 90000),
            new Transaction("C202", 14000, base + 130000)
        )
        .keyBy(tx -> tx.customerId)
        .process(new Detector())
        .print();

        env.execute("Fraud Detection Job");
    }
}
```

### 2. Install Dependencies / Build Tools
Ensure Maven and JDK 17 are installed (`mvn -version`, `java -version`).

### 3. Run Commands
*(Do not use `mvn exec:java` because of classloader issues; use `dependency:copy-dependencies` and `java -cp`):*
```powershell
cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P6"
mvn clean package dependency:copy-dependencies
java -cp "target\flink-practical-1.0-SNAPSHOT.jar;target\dependency\*" FraudDetectionJob
```

### 4. Expected Output
```
9> FRAUD ALERT: C101, Previous=₹15000.0, Current=₹20000.0, Gap=35 seconds
9> FRAUD ALERT: C101, Previous=₹20000.0, Current=₹18000.0, Gap=45 seconds
```

---

## Practical 7: Watermarks & Late Data Handling with Apache Flink (PyFlink)

### 1. Files to Create
Create directory `P7` and file `P7/latedata.py`:

```python
import time
from pyflink.common import Types, WatermarkStrategy, Duration
from pyflink.common.time import Time
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.output_tag import OutputTag
from pyflink.datastream.window import TumblingEventTimeWindows

class MyTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, event, record_timestamp):
        return event[1]

env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)

# Delivery events: (event name, timestamp in milliseconds)
events = [
    ("Picked up", 1000),
    ("At restaurant", 8000),
    ("Delivered", 22000),
    ("Order confirmed", 3000)
]

data = env.from_collection(
    events,
    type_info=Types.TUPLE([Types.STRING(), Types.LONG()])
)

# Allow 2 seconds of out-of-orderness
watermarks = WatermarkStrategy.for_bounded_out_of_orderness(
    Duration.of_seconds(2)
).with_timestamp_assigner(MyTimestampAssigner())

def delay(event):
    time.sleep(5 if event[0] == "Order confirmed" else 1)
    return event

data = data.map(
    delay,
    output_type=Types.TUPLE([Types.STRING(), Types.LONG()])
)

stream = data.assign_timestamps_and_watermarks(watermarks)

late_tag = OutputTag("late", Types.TUPLE([Types.STRING(), Types.LONG()]))

windowed = (
    stream
    .key_by(lambda x: "delivery", key_type=Types.STRING())
    .window(TumblingEventTimeWindows.of(Time.seconds(10)))
    .side_output_late_data(late_tag)
)

result = windowed.reduce(
    lambda a, b: (a[0] + " + " + b[0], max(a[1], b[1])),
    output_type=Types.TUPLE([Types.STRING(), Types.LONG()])
)

result.print("MAIN OUTPUT")
result.get_side_output(late_tag).print("LATE EVENTS")

env.execute("Food Delivery Late Data")
```

### 2. Install Dependencies
```powershell
pip install apache-flink
```

### 3. Run Commands
```powershell
cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P7"
python latedata.py
```

### 4. Expected Output
```
MAIN OUTPUT> (Picked up + At restaurant + Order confirmed,8000)
MAIN OUTPUT> (Delivered,22000)
```

---

## Practical 8: Graph Processing & PageRank using Spark GraphX (Scala / SBT)

### 1. Files to Create
Create directory `P8`, subfolders `P8/data` and `P8/src/main/scala`, and create the following 4 files:

#### File: `P8/build.sbt`
```scala
name := "GraphXPageRank"
version := "1.0"
scalaVersion := "2.12.18"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.7",
  "org.apache.spark" %% "spark-sql" % "3.5.7",
  "org.apache.spark" %% "spark-graphx" % "3.5.7"
)

Compile / run / fork := true

Compile / run / javaOptions ++= Seq(
  "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
  "--add-opens=java.base/java.io=ALL-UNNAMED",
  "--add-opens=java.base/java.net=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
  "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
  "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED"
)
```

#### File: `P8/data/vertices.csv`
```csv
id,name
1,Alice
2,Bob
3,Charlie
4,David
5,Emma
6,Frank
```

#### File: `P8/data/edges.csv`
```csv
src,dst
1,2
1,3
2,3
2,4
3,1
3,4
4,5
5,3
5,6
6,1
```

#### File: `P8/src/main/scala/GraphXPageRank.scala`
```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphXPageRank {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("GraphX PageRank")
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    println("Spark Started Successfully")

    val users: RDD[(VertexId, String)] =
      sc.textFile("data/vertices.csv")
        .filter(!_.startsWith("id,"))
        .map { line =>
          val fields = line.split(",")
          (fields(0).toLong, fields(1))
        }

    val relationships: RDD[Edge[Int]] =
      sc.textFile("data/edges.csv")
        .filter(!_.startsWith("src,"))
        .map { line =>
          val fields = line.split(",")
          Edge(fields(0).toLong, fields(1).toLong, 1)
        }

    val graph = Graph(users, relationships)

    println("Graph Created Successfully")
    println("Total Users: " + graph.numVertices)
    println("Total Relationships: " + graph.numEdges)

    val ranks = graph.pageRank(0.0001).vertices

    val result = users.join(ranks)
      .map { case (id, (name, rank)) => (name, rank) }
      .sortBy(_._2, ascending = false)

    println("PageRank Results")
    result.collect().foreach { case (name, rank) => println(name + " : " + rank) }

    println("Most Influential User")
    println(result.first())

    spark.stop()
  }
}
```

### 2. Install Dependencies / Build Tools
Ensure `sbt` and JDK 17 are installed (`sbt -version`, `java -version`).

### 3. Run Commands
```powershell
cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P8"
sbt run
```

### 4. Expected Output
```
Total Users: 6
Total Relationships: 10
PageRank Results
Charlie : 1.4093528428285806
Alice : 1.2494627952608812
David : 1.0384527360626896
Emma : 1.0326750168080894
Bob : 0.6810677646742912
Frank : 0.588988844365468
Most Influential User
(Charlie,1.4093528428285806)
```

---

## Practical 9: Federated SQL Queries with Trino & MySQL

### 1. Files to Create
Create directory `P9`, subfolders `P9/mysql` and `P9/etc/catalog`, and create the following 3 files:

#### File: `P9/docker-compose.yml`
```yaml
services:
  mysql:
    image: mysql:8.0
    container_name: trino_mysql
    environment:
      MYSQL_ROOT_PASSWORD: root123
      MYSQL_DATABASE: university
    ports:
      - "3307:3306"
    volumes:
      - ./mysql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro

  trino:
    image: trinodb/trino:latest
    container_name: trino_coordinator
    ports:
      - "8082:8080"
    volumes:
      - ./etc/catalog:/etc/trino/catalog
    depends_on:
      - mysql
```

#### File: `P9/mysql/init.sql`
```sql
CREATE TABLE university.students (
    student_id INT PRIMARY KEY,
    student_name VARCHAR(100),
    department VARCHAR(100)
);

INSERT INTO university.students VALUES
(101, 'Aarav', 'Data Science'),
(102, 'Diya', 'Computer Science'),
(103, 'Rohan', 'Data Science'),
(104, 'Meera', 'Computer Science');

CREATE USER IF NOT EXISTS 'trino_user'@'%' IDENTIFIED BY 'trino123';
GRANT SELECT ON university.* TO 'trino_user'@'%';
```

#### File: `P9/etc/catalog/mysql.properties`
```properties
connector.name=mysql
connection-url=jdbc:mysql://mysql:3306
connection-user=trino_user
connection-password=trino123
```

### 2. Install Dependencies
Ensure Docker Desktop is running.

### 3. Run Commands
1. **Start the Containers**:
   ```powershell
   cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P9"
   docker compose up -d
   ```
2. **Wait ~20 seconds for MySQL and Trino initialization.**
3. **Query via Trino Coordinator**:
   ```powershell
   docker exec -it trino_coordinator trino --execute "SELECT * FROM mysql.university.students;"
   ```

### 4. Expected Output
```
 student_id | student_name |   department     
------------+--------------+------------------
        101 | Aarav        | Data Science     
        102 | Diya         | Computer Science 
        103 | Rohan        | Data Science     
        104 | Meera        | Computer Science 
```
Trino Web UI is available at `http://localhost:8082`.

### 5. Cleanup
```powershell
docker compose down
```

---

## Practical 10: End-to-End Real-Time Analytics (Kafka -> ClickHouse -> Grafana)

### 1. Files to Create
Create directory `P10` and the following 5 files:

#### File: `P10/docker-compose.yml`
```yaml
services:
  kafka:
    image: apache/kafka:3.9.1
    ports:
      - "9093:9093"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
      KAFKA_NUM_PARTITIONS: 1

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports:
      - "8123:8123"
      - "9000:9000"
    environment:
      CLICKHOUSE_SKIP_USER_SETUP: 1

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_INSTALL_PLUGINS: grafana-clickhouse-datasource
    depends_on:
      - clickhouse

  producer:
    build: .
    command: python -u producer.py
    depends_on:
      - kafka

  consumer:
    build: .
    command: python -u consumer.py
    depends_on:
      - kafka
      - clickhouse
```

#### File: `P10/Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY producer.py .
COPY consumer.py .

CMD ["python", "-u", "producer.py"]
```

#### File: `P10/requirements.txt`
```
kafka-python==2.2.15
clickhouse-connect==1.11.0
```

#### File: `P10/producer.py`
```python
import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

while True:
    try:
        producer = KafkaProducer(
            bootstrap_servers="kafka:9092",
            value_serializer=lambda x: json.dumps(x).encode()
        )
        print("Kafka Connected!")
        break
    except NoBrokersAvailable:
        print("Waiting for Kafka...")
        time.sleep(3)

cities = ["Mumbai", "Pune", "Nashik", "Nagpur"]

while True:
    data = {
        "event_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": random.choice(cities),
        "pm25": round(random.uniform(10, 180), 2),
        "pm10": round(random.uniform(20, 250), 2),
        "temperature": round(random.uniform(24, 38), 2)
    }
    producer.send("air_quality", data)
    producer.flush()
    print("Sent:", data)
    time.sleep(2)
```

#### File: `P10/consumer.py`
```python
import json
import time
import clickhouse_connect
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

while True:
    try:
        db = clickhouse_connect.get_client(host="clickhouse", port=8123)
        db.command("""
            CREATE TABLE IF NOT EXISTS default.air_quality (
                event_time DateTime,
                city String,
                pm25 Float32,
                pm10 Float32,
                temperature Float32
            ) ENGINE = MergeTree()
            ORDER BY (city, event_time)
        """)
        print("ClickHouse Connected!")
        break
    except Exception as e:
        print("Waiting for ClickHouse...", e)
        time.sleep(3)

while True:
    try:
        consumer = KafkaConsumer(
            "air_quality",
            bootstrap_servers="kafka:9092",
            auto_offset_reset="earliest",
            value_deserializer=lambda x: json.loads(x.decode())
        )
        print("Kafka Connected!")
        break
    except NoBrokersAvailable:
        print("Waiting for Kafka...")
        time.sleep(3)

for msg in consumer:
    d = msg.value
    db.insert("air_quality", [[
        d["event_time"], d["city"], d["pm25"],
        d["pm10"], d["temperature"]
    ]], column_names=[
        "event_time", "city", "pm25", "pm10", "temperature"
    ])
    print("Inserted:", d)
```

### 2. Install Dependencies
Ensure Docker Desktop is running.

### 3. Run Commands
1. **Build and Run All Services**:
   ```powershell
   cd "c:\Users\halda\OneDrive\Desktop\Practical EXAM\BigDiddy\P10"
   docker compose up --build -d
   ```
2. **Monitor Producer Stream**:
   ```powershell
   docker compose logs -f producer
   ```
3. **Monitor Consumer Inserts**:
   ```powershell
   docker compose logs -f consumer
   ```
4. **Query ClickHouse Database Directly**:
   ```powershell
   docker exec -it p10-clickhouse-1 clickhouse-client --query "SELECT count(), city, round(avg(pm25),2), round(avg(temperature),2) FROM default.air_quality GROUP BY city;"
   ```
5. **Open Grafana**:
   - URL: `http://localhost:3000` (User: `admin`, Pass: `admin`)
   - Add ClickHouse datasource (`Server: clickhouse`, `Port: 8123`).

### 4. Expected Output
- Continuous live stream of simulated air quality sensor events ingested into Kafka, consumed, and stored in ClickHouse columnar storage.

### 5. Cleanup
```powershell
docker compose down
```
