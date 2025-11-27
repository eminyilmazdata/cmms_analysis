"""
Synthetic CMMS Data Generator

Generates realistic synthetic data for equipment failures and maintenance tasks
to enable analysis of preventive maintenance effectiveness.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random


class CMMSDataGenerator:
    """
    Generates synthetic CMMS data including:
    - Equipment information
    - Failure events
    - Preventive maintenance tasks
    - Corrective maintenance tasks
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.equipment_types = [
            'Pump', 'Compressor', 'Motor', 'Generator', 'Turbine',
            'Conveyor', 'Valve', 'Heat Exchanger', 'Boiler', 'Chiller'
        ]
        
        self.failure_modes = [
            'Bearing Failure', 'Seal Leak', 'Overheating', 'Vibration',
            'Corrosion', 'Wear', 'Electrical Fault', 'Lubrication Issue',
            'Alignment Problem', 'Crack/Fracture'
        ]
        
        self.maintenance_types = [
            'Inspection', 'Lubrication', 'Calibration', 'Cleaning',
            'Replacement', 'Adjustment', 'Testing', 'Alignment'
        ]
    
    def generate_equipment(self, n_equipment: int = 10) -> pd.DataFrame:
        """
        Generate equipment master data.
        
        Parameters:
        -----------
        n_equipment : int
            Number of equipment items to generate
            
        Returns:
        --------
        equipment_df : pd.DataFrame
            DataFrame with equipment information
        """
        equipment_data = []
        
        for i in range(n_equipment):
            equipment_id = f"EQ-{i+1:04d}"
            equipment_type = np.random.choice(self.equipment_types)
            
            # Installation date (random date in the past 5 years)
            days_ago = np.random.randint(365, 5 * 365)
            installation_date = datetime.now() - timedelta(days=days_ago)
            
            # Criticality level
            criticality = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
            
            # Operating hours per day (affects failure rate)
            operating_hours_per_day = np.random.uniform(8, 24)
            
            equipment_data.append({
                'equipment_id': equipment_id,
                'equipment_type': equipment_type,
                'installation_date': installation_date,
                'criticality': criticality,
                'operating_hours_per_day': operating_hours_per_day,
                'manufacturer': f"Manufacturer-{np.random.randint(1, 6)}",
                'model': f"Model-{np.random.randint(100, 999)}"
            })
        
        return pd.DataFrame(equipment_data)
    
    def generate_failures(self, equipment_df: pd.DataFrame, 
                         start_date: datetime,
                         end_date: datetime,
                         with_pm_effect: bool = True,
                         pm_effectiveness: float = 0.3) -> pd.DataFrame:
        """
        Generate failure events for equipment.
        
        Parameters:
        -----------
        equipment_df : pd.DataFrame
            Equipment master data
        start_date : datetime
            Start date for failure generation
        end_date : datetime
            End date for failure generation
        with_pm_effect : bool
            Whether preventive maintenance affects failure rate
        pm_effectiveness : float
            Effectiveness of PM (0-1, how much it reduces failure rate)
            
        Returns:
        --------
        failures_df : pd.DataFrame
            DataFrame with failure events
        """
        failures = []
        current_date = start_date
        
        # Generate PM schedule (will be used if with_pm_effect is True)
        pm_schedule = self._generate_pm_schedule(equipment_df, start_date, end_date)
        
        for _, eq in equipment_df.iterrows():
            equipment_id = eq['equipment_id']
            installation_date = eq['installation_date']
            
            # Base failure rate (failures per 1000 operating hours)
            # Higher for older equipment and critical equipment
            age_factor = (datetime.now() - installation_date).days / 365.0
            criticality_factor = {'Low': 0.5, 'Medium': 1.0, 'High': 1.5}[eq['criticality']]
            
            base_failure_rate = 0.5 + 0.3 * age_factor + 0.2 * criticality_factor
            
            # Operating hours per day
            hours_per_day = eq['operating_hours_per_day']
            
            # Time between failures (in days) - Weibull distribution
            # Shape parameter controls failure pattern
            shape = 1.5  # Increasing failure rate over time
            scale = 30 / base_failure_rate  # Scale based on failure rate
            
            current_time = max(start_date, installation_date)
            failure_count = 0
            
            while current_time < end_date:
                # Calculate time to next failure
                # Use Weibull distribution for realistic failure patterns
                time_to_failure_days = np.random.weibull(shape) * scale
                
                # Adjust for PM effect if enabled
                if with_pm_effect:
                    # Check if PM was performed recently
                    recent_pms = pm_schedule[
                        (pm_schedule['equipment_id'] == equipment_id) &
                        (pm_schedule['task_date'] <= current_time) &
                        (pm_schedule['task_date'] > current_time - timedelta(days=30))
                    ]
                    
                    if len(recent_pms) > 0:
                        # PM reduces failure rate
                        time_to_failure_days *= (1 + pm_effectiveness)
                
                current_time += timedelta(days=time_to_failure_days)
                
                if current_time > end_date:
                    break
                
                failure_count += 1
                
                # Failure mode
                failure_mode = np.random.choice(self.failure_modes)
                
                # Severity (affects downtime)
                severity = np.random.choice(['Minor', 'Moderate', 'Major', 'Critical'], 
                                           p=[0.4, 0.3, 0.2, 0.1])
                
                # Downtime (hours) based on severity
                downtime_hours = {
                    'Minor': np.random.uniform(1, 4),
                    'Moderate': np.random.uniform(4, 12),
                    'Major': np.random.uniform(12, 48),
                    'Critical': np.random.uniform(48, 168)
                }[severity]
                
                failures.append({
                    'failure_id': f"FAIL-{equipment_id}-{failure_count:03d}",
                    'equipment_id': equipment_id,
                    'failure_date': current_time,
                    'failure_mode': failure_mode,
                    'severity': severity,
                    'downtime_hours': downtime_hours,
                    'operating_hours_at_failure': failure_count * hours_per_day * time_to_failure_days
                })
        
        failures_df = pd.DataFrame(failures)
        if len(failures_df) > 0:
            failures_df = failures_df.sort_values('failure_date').reset_index(drop=True)
        
        return failures_df
    
    def _generate_pm_schedule(self, equipment_df: pd.DataFrame,
                             start_date: datetime,
                             end_date: datetime) -> pd.DataFrame:
        """
        Generate preventive maintenance schedule.
        
        Parameters:
        -----------
        equipment_df : pd.DataFrame
            Equipment master data
        start_date : datetime
            Start date
        end_date : datetime
            End date
            
        Returns:
        --------
        pm_schedule_df : pd.DataFrame
            DataFrame with PM schedule
        """
        pm_tasks = []
        
        for _, eq in equipment_df.iterrows():
            equipment_id = eq['equipment_id']
            
            # PM frequency (days between PM tasks)
            # More critical equipment gets more frequent PM
            criticality_pm_freq = {'Low': 90, 'Medium': 60, 'High': 30}[eq['criticality']]
            pm_frequency_days = criticality_pm_freq + np.random.randint(-10, 10)
            
            current_date = start_date
            
            while current_date < end_date:
                # PM task type
                pm_type = np.random.choice(self.maintenance_types)
                
                # Duration (hours)
                duration_hours = np.random.uniform(1, 8)
                
                pm_tasks.append({
                    'task_id': f"PM-{equipment_id}-{len(pm_tasks)+1:04d}",
                    'equipment_id': equipment_id,
                    'task_date': current_date,
                    'task_type': pm_type,
                    'maintenance_type': 'Preventive',
                    'duration_hours': duration_hours,
                    'technician_id': f"TECH-{np.random.randint(1, 11):02d}"
                })
                
                current_date += timedelta(days=pm_frequency_days)
        
        return pd.DataFrame(pm_tasks)
    
    def generate_maintenance_tasks(self, equipment_df: pd.DataFrame,
                                  failures_df: pd.DataFrame,
                                  start_date: datetime,
                                  end_date: datetime) -> pd.DataFrame:
        """
        Generate maintenance tasks (both preventive and corrective).
        
        Parameters:
        -----------
        equipment_df : pd.DataFrame
            Equipment master data
        failures_df : pd.DataFrame
            Failure events
        start_date : datetime
            Start date
        end_date : datetime
            End date
            
        Returns:
        --------
        maintenance_df : pd.DataFrame
            DataFrame with all maintenance tasks
        """
        # Generate preventive maintenance
        pm_df = self._generate_pm_schedule(equipment_df, start_date, end_date)
        
        # Generate corrective maintenance from failures
        cm_tasks = []
        
        for _, failure in failures_df.iterrows():
            # Corrective maintenance happens after failure
            # Add some delay (time to respond and repair)
            response_time_hours = np.random.uniform(2, 24)
            repair_date = failure['failure_date'] + timedelta(hours=response_time_hours)
            
            # Repair duration based on severity
            repair_duration_hours = failure['downtime_hours'] * np.random.uniform(0.8, 1.2)
            
            cm_tasks.append({
                'task_id': f"CM-{failure['failure_id']}",
                'equipment_id': failure['equipment_id'],
                'task_date': repair_date,
                'task_type': 'Repair',
                'maintenance_type': 'Corrective',
                'duration_hours': repair_duration_hours,
                'failure_id': failure['failure_id'],
                'technician_id': f"TECH-{np.random.randint(1, 11):02d}"
            })
        
        cm_df = pd.DataFrame(cm_tasks)
        
        # Combine PM and CM
        all_maintenance = pd.concat([pm_df, cm_df], ignore_index=True)
        all_maintenance = all_maintenance.sort_values('task_date').reset_index(drop=True)
        
        return all_maintenance
    
    def generate_complete_dataset(self, n_equipment: int = 10,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None,
                                  with_pm_effect: bool = True,
                                  pm_effectiveness: float = 0.3) -> Dict[str, pd.DataFrame]:
        """
        Generate complete CMMS dataset.
        
        Parameters:
        -----------
        n_equipment : int
            Number of equipment items
        start_date : datetime, optional
            Start date (default: 2 years ago)
        end_date : datetime, optional
            End date (default: today)
        with_pm_effect : bool
            Whether PM affects failure rate
        pm_effectiveness : float
            PM effectiveness (0-1)
            
        Returns:
        --------
        dataset : dict
            Dictionary with 'equipment', 'failures', and 'maintenance' DataFrames
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=2 * 365)
        if end_date is None:
            end_date = datetime.now()
        
        equipment_df = self.generate_equipment(n_equipment)
        failures_df = self.generate_failures(
            equipment_df, start_date, end_date, with_pm_effect, pm_effectiveness
        )
        maintenance_df = self.generate_maintenance_tasks(
            equipment_df, failures_df, start_date, end_date
        )
        
        return {
            'equipment': equipment_df,
            'failures': failures_df,
            'maintenance': maintenance_df
        }

