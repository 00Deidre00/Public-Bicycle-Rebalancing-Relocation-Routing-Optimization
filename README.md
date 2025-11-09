# 🚲 Public Bicycle Rebalancing, Relocation & Routing Optimization

## 📘 Project Overview
This project focuses on optimizing **public bicycle rebalancing and relocation**.  
It aims to minimize station inventory imbalances by optimizing truck routes and the pickup/drop-off operations.

### Objectives
- Minimize inventory imbalance across bicycle stations  
- Optimize truck travel and operation time  
- Consider real-world constraints such as time windows, truck capacity, and inventory bounds  

---

## 🧩 Repository Structure

/
├── bound/
│ Contains files used to determine the safety stock levels for each station.
│ Example: time-dependent lower/upper bounds for inventory, optimal stock ranges.

├── model/
│ Stores trained predictive models (.pkl files) for each station.
│ Example: demand prediction models or inventory change models (provided as samples).

├── src/
│ Contains the core code of the project.
│ Includes scripts for route optimization, relocation simulation, constraint application, and model execution.

├── RE;PATH/
│ Folder for app integration and visualization of results.
│ Example: route output files and data conversion scripts for the app.

└── README.md
Provides an overview of the project, installation instructions, and model structure.
