"""
Test script to compare damaged vs undamaged images
Shows how color balance and sharpening detection features differ
"""

from src.ml.feature_extractor import extract_ml_features
import os

# Test images
damaged = 'data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged/1.png'
undamaged = 'data/raw/AI_for_Art_Restoration_2/paired_dataset_art/undamaged/1.png'

print('COMPARISON: Damaged vs Undamaged Image')
print('='*80)

# Extract features
f_damaged, names = extract_ml_features(damaged)
f_undamaged, _ = extract_ml_features(undamaged)

print('\nFeature                   Damaged      Undamaged    Difference')
print('-'*80)

for name, d_val, u_val in zip(names, f_damaged, f_undamaged):
    diff = d_val - u_val
    marker = ' ***' if abs(diff) > 0.2 else ''
    print(f'{name:<25} {d_val:>8.4f}    {u_val:>8.4f}    {diff:>8.4f}{marker}')

print('='*80)
print('\n🎯 KEY DETECTION FEATURES:')
print('-'*80)
print(f'  Color Balance Need:')
print(f'    Damaged   : {f_damaged[-2]:.4f} (higher = more correction needed)')
print(f'    Undamaged : {f_undamaged[-2]:.4f}')
print(f'    Difference: {f_damaged[-2] - f_undamaged[-2]:.4f}')
print()
print(f'  Sharpening Need:')
print(f'    Damaged   : {f_damaged[-1]:.4f} (higher = more sharpening needed)')
print(f'    Undamaged : {f_undamaged[-1]:.4f}')
print(f'    Difference: {f_damaged[-1] - f_undamaged[-1]:.4f}')
print('='*80)

# Test on multiple images
print('\n📊 TESTING ON MULTIPLE IMAGE PAIRS:')
print('='*80)

damaged_dir = 'data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged'
undamaged_dir = 'data/raw/AI_for_Art_Restoration_2/paired_dataset_art/undamaged'

if os.path.exists(damaged_dir) and os.path.exists(undamaged_dir):
    files = sorted(os.listdir(damaged_dir))[:5]  # Test first 5 images
    
    print('\nImage    Color Balance Need       Sharpening Need')
    print('         Damaged  Undamaged       Damaged  Undamaged')
    print('-'*80)
    
    for file in files:
        try:
            d_path = os.path.join(damaged_dir, file)
            u_path = os.path.join(undamaged_dir, file)
            
            if os.path.exists(d_path) and os.path.exists(u_path):
                f_d, _ = extract_ml_features(d_path)
                f_u, _ = extract_ml_features(u_path)
                
                print(f'{file:<8} {f_d[-2]:>7.4f}   {f_u[-2]:>7.4f}        {f_d[-1]:>7.4f}   {f_u[-1]:>7.4f}')
        except Exception as e:
            print(f'{file:<8} Error: {e}')
    
    print('='*80)
    print('\n✅ Detection features successfully added to ML model!')
    print('   These features help the model decide when to apply:')
    print('   - Color Balance Correction')
    print('   - Sharpening Enhancement')
