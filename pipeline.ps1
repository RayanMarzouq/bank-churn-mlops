# Fichier: pipeline.ps1
# Orchestration MLOps : Détection de Dérive et Ré-entraînement Conditionnel

# --- 1. CONFIGURATION ---
$driftScript = "python detect_drift.py"
$trainScript = "python train_model.py"

Write-Host "============================================="
Write-Host "🚀 DÉBUT DE LA PIPELINE MLOPS"
Write-Host "============================================="

# --- 2. ÉTAPE DE MONITORING (DÉTECTION DE DÉRIVE) ---
Write-Host "`n---> [1/2] DÉMARRAGE DU MONITORING (detect_drift.py)"
Write-Host "---------------------------------------------"

Invoke-Expression $driftScript
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "✅ STATUT: Aucune dérive majeure détectée. Le modèle actuel est maintenu."
    Write-Host "============================================="
    exit 0
} elseif ($exitCode -eq 1) {
    Write-Host "🚨 ALERTE: Dérive Majeure Détectée (Code $exitCode)."

    # --- 3. ÉTAPE DE RÉ-ENTRAÎNEMENT (MLflow Tracking) ---
    Write-Host "`n---> [2/2] DÉCLENCHEMENT DU RÉ-ENTRAÎNEMENT (train_model.py)"
    Write-Host "---------------------------------------------"

    Invoke-Expression $trainScript

    Write-Host "`n✅ STATUT: Le ré-entraînement est terminé."
    Write-Host "Le nouveau modèle est enregistré dans MLflow (Model Registry)."
    Write-Host "============================================="
    exit 0
} else {
    Write-Host "❌ ERREUR: Erreur inattendue dans la détection de dérive (Code $exitCode)."
    Write-Host "============================================="
    exit 1
}