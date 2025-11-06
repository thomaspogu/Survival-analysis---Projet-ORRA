# Script R pour tracer la courbe de Kaplan-Meier avec mlr3proba
# Utilise le même dataset que k_m_estimator_skicit.ipynb (veterans_lung_cancer)

# Installer les packages si nécessaire:
# install.packages(c("survival", "mlr3", "mlr3proba", "ggplot2"))

# Charger les bibliothèques nécessaires
library(survival)
library(mlr3)
library(mlr3proba)
library(ggplot2)

# Charger le dataset veterans_lung_cancer
# Ce dataset est disponible dans le package survival
# Utilisation de survival::veteran pour éviter l'avertissement
veteran <- survival::veteran

# Afficher les premières lignes et la structure des données
cat("Structure du dataset:\n")
str(veteran)
cat("\nPremières lignes:\n")
head(veteran)

# Le dataset veteran contient:
# - time: temps de survie en jours
# - status: statut (1 = décès, 0 = censure)
# - autres variables: trt, celltype, prior, karno, diagtime, age, etc.

# Statistiques descriptives
cat("\nNombre total de patients:", length(veteran$time), "\n")
cat("Événements observés (décès):", sum(veteran$status == 1), "\n")
cat("Censures:", sum(veteran$status == 0), "\n")

# ============================================================================
# VERSION AVEC mlr3proba
# ============================================================================

cat("\n=== Estimation de Kaplan-Meier avec mlr3proba ===\n\n")

# Créer une tâche de survie avec mlr3proba
# Note: mlr3proba attend que l'événement soit TRUE/FALSE ou 1/0
# Le dataset veteran utilise status = 1 pour décès, ce qui est compatible
task_km <- TaskSurv$new(
  id = "veteran_km", 
  backend = veteran, 
  time = "time", 
  event = "status"
)

# Afficher les informations sur la tâche
cat("Informations sur la tâche de survie:\n")
print(task_km)

# Créer et entraîner l'estimateur de Kaplan-Meier
kaplan <- lrn("surv.kaplan")
kaplan$train(task_km)

# Obtenir les prédictions
pred <- kaplan$predict(task_km)

# Extraire la distribution de survie
# mlr3proba stocke les prédictions dans un objet de distribution
surv_dist <- pred$distr

# Extraire les temps et probabilités de survie
# Pour surv.kaplan, toutes les prédictions sont identiques (courbe globale)
# On extrait les données de la première prédiction

# Méthode d'extraction: utiliser la méthode survival() de la distribution
# Cette méthode retourne la fonction de survie pour un temps donné
# On doit d'abord obtenir les temps uniques, puis calculer la survie pour chaque temps

# Obtenir les temps uniques depuis les données
times_unique <- sort(unique(veteran$time))

# Calculer la probabilité de survie pour chaque temps
# Pour surv.kaplan, on peut utiliser la méthode survival() de la distribution
# Prendre la première valeur (toutes identiques pour KM)
surv_probs <- sapply(times_unique, function(t) {
  surv_dist$survival(t)[1]
})

# Créer un dataframe pour le tracé
df_km <- data.frame(
  time = times_unique,
  surv = surv_probs
)

# Tracer la courbe de Kaplan-Meier avec ggplot2
p <- ggplot(df_km, aes(x = time, y = surv)) +
  geom_step(color = "darkred", linewidth = 1.2) +
  geom_ribbon(aes(ymin = 0, ymax = surv), alpha = 0.3, fill = "lightcoral") +
  labs(
    title = "Courbe de Kaplan-Meier - Dataset Veterans Lung Cancer (mlr3proba)",
    x = "Temps (jours)",
    y = "Probabilité de survie"
  ) +
  theme_bw() +
  theme(plot.title = element_text(face = "bold", size = 14)) +
  scale_y_continuous(limits = c(0, 1))

print(p)

# Afficher des statistiques
cat("\nStatistiques avec mlr3proba:\n")
cat("Nombre de points temporels:", length(times_unique), "\n")
cat("Probabilité de survie initiale:", round(surv_probs[1], 4), "\n")
cat("Probabilité de survie finale:", round(surv_probs[length(surv_probs)], 4), "\n")
cat("Temps minimum:", min(times_unique), "jours\n")
cat("Temps maximum:", max(times_unique), "jours\n")

# Calculer le temps médian de survie (probabilité = 0.5)
median_idx <- which(surv_probs <= 0.5)[1]
if (!is.na(median_idx)) {
  median_time <- times_unique[median_idx]
  cat("Temps médian de survie:", round(median_time, 0), "jours\n")
} else {
  cat("Temps médian de survie: Non atteint (>", max(times_unique), "jours)\n")
}
