import sqlite3

conn = sqlite3.connect("grievance.db")
c = conn.cursor()

# delete old admin (if partial entry exists)
c.execute("DELETE FROM users WHERE username='admin'")

# insert fresh admin
c.execute("INSERT INTO users VALUES ('admin','admin123','admin')")

conn.commit()
print("Admin reset successfully!")

conn.close()