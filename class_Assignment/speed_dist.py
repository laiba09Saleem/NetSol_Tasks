import numpy as np


# GIVEN DATA (From Google Maps)

road_distance = 5.9  # km (actual road distance)
car_time_min = 14
bike_time_min = 12

# Convert minutes to hours
car_time_hr = car_time_min / 60
bike_time_hr = bike_time_min / 60


#  SPEED USING ROAD DISTANCE

car_speed = road_distance / car_time_hr
bike_speed = road_distance / bike_time_hr

print("Road Distance:", road_distance, "km")
print("Car Speed:", car_speed, "km/h")
print("Bike Speed:", bike_speed, "km/h")


# DISPLACEMENT (STRAIGHT LINE)

# Assume coordinates 

LGU = np.array([0, 0])
Home = np.array([4.8, 2.5])   # Not straight horizontal now

# Displacement Vector
displacement_vector = Home - LGU

# Magnitude of Displacement
displacement = np.linalg.norm(displacement_vector)

print("\nDisplacement Vector:", displacement_vector)
print("Displacement (Straight Line):", displacement, "km")


# CONSTANT TIME FUNCTION MODELING

constant_time = 0.25  # hours

speed_constant = road_distance / constant_time
displacement_constant = displacement  # displacement stays fixed between two points

print("\nSpeed (Constant Time):", speed_constant, "km/h")
print("Displacement (Constant Time):", displacement_constant, "km")


# VARIABLE TIME FUNCTION MODELING

time_values = np.array([0.15, 0.20, 0.25, 0.30])

speed_function = road_distance / time_values

# Distance covered in each case
distance_function = speed_function * time_values

print("\nTime Values:", time_values)
print("Speed Function Values:", speed_function)
print("Distance Covered:", distance_function)
print("Displacement (remains constant):", displacement)