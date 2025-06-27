
CREATE DATABASE safety_monitoring;
USE safety_monitoring;

CREATE TABLE alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    alert_time DATETIME NOT NULL,
    coord_x INT NOT NULL,
    coord_y INT NOT NULL
);
