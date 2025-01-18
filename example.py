from core.doc_retrievaler import DocRetrievaler
from core.doc_trainer import Trainer
from core.reasoner import Reasoner

query = "I want to import my data using 'insert' to create a table with InnoDB. The table named `customer_orders` with the following structure:\n\n```sql\nCREATE TABLE customer_orders (\n    id INT PRIMARY KEY AUTO_INCREMENT,\n    customer_name VARCHAR(100),\n    order_amount DECIMAL(10, 2),\n    order_date DATE\n);\n```\n\nTo populate this table efficiently, I've created a stored procedure `populate_customer_orders`:\n\n```sql\nDELIMITER //\n\nCREATE PROCEDURE populate_orders(IN num_orders INT)\nBEGIN\n    DECLARE i INT DEFAULT 1;\n    DECLARE customer_id_val INT;\n    DECLARE product_id_val INT;\n\n    WHILE i <= num_orders DO\n        SELECT id INTO customer_id_val FROM customers ORDER BY RAND() LIMIT 1;\n        SELECT id INTO product_id_val FROM products ORDER BY RAND() LIMIT 1;\n        INSERT INTO orders (customer_id, product_id, order_date)\n        VALUES (customer_id_val, product_id_val, DATE_ADD('2024-01-01', INTERVAL FLOOR(RAND() * 365) DAY));\n        \n        SET i = i + 1;\n    END WHILE;\nEND //\n\nDELIMITER ;\n``` \n\nWhen I execute `CALL populate_orders(1000000);` to insert 1 million records, it takes an unexpectedly long time (approximately 15 minutes). However, importing the same data via a CSV file into MySQL completes within a few minutes. \n\nI suspect there might be inefficiencies in how MySQL handles transactions or buffer management during procedural insertion. What MySQL parameters should I adjust to improve the performance of my stored procedure? Are there any best practices or alternative methods I should consider to expedite large-scale data insertion in MySQL? \n\nI'm looking forward to any insights or suggestions on optimizing MySQL performance for bulk data insertion. Thank you!"
data_path='DOWNLOAD_DATA_IN_THIS_PATH'
dataset = 'mysql_so'
trainer = Trainer(dataset=dataset, data_path=data_path)
trainer.train()

doc_retrievaler = DocRetrievaler(dataset, index_path=rf'{data_path}/vb.index',data_path=data_path)
doc_retrievaler.build_index()
retrieved_docs = doc_retrievaler.search(query)[0]

reasoner = Reasoner(dataset, data_path, gpt_model_version='gpt-3.5-turbo')
prompt = reasoner.generate_prompt(query, retrieved_docs)
print('prompt', prompt)
response = reasoner.apply(prompt)

print(response)

