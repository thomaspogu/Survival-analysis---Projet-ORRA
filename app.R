library(shiny)
library(shinydashboard)
library(survival)
library(mlr3)
library(mlr3proba)
library(ggplot2)

# Charger le dataset (en dur dans le code pour l'instant)
veteran <- survival::veteran

# Interface utilisateur
ui <- dashboardPage(
  dashboardHeader(title = "Analyse de Survie - Kaplan-Meier & Métriques"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Visualisation", tabName = "visualization", icon = icon("chart-line")),
      menuItem("Métriques", tabName = "metrics", icon = icon("calculator"))
    ),
    hr(),
    h4("Paramètres du modèle", style = "padding-left: 15px;"),
    sliderInput("train_ratio", "Ratio Train/Test", 
                min = 0.5, max = 0.95, value = 0.8, step = 0.05),
    numericInput("seed", "Seed aléatoire", value = 42, min = 1),
    hr(),
    actionButton("calculate", "Calculer", 
                 class = "btn-primary", 
                 style = "margin-left: 15px; margin-right: 15px; width: calc(100% - 30px);")
  ),
  dashboardBody(
    tabItems(
      # Onglet Visualisation
      tabItem(tabName = "visualization",
        fluidRow(
          box(title = "Courbe de Kaplan-Meier", status = "primary", 
              solidHeader = TRUE, width = 12,
              plotOutput("km_plot", height = "500px")
          )
        ),
        fluidRow(
          box(title = "Informations sur le dataset", status = "info", 
              solidHeader = TRUE, width = 12,
              verbatimTextOutput("dataset_info")
          )
        )
      ),
      # Onglet Métriques
      tabItem(tabName = "metrics",
        fluidRow(
          box(title = "C-index", status = "success", 
              solidHeader = TRUE, width = 12,
              verbatimTextOutput("cindex_output"),
              h5("Paramètres utilisés:", style = "margin-top: 15px; font-weight: bold;"),
              verbatimTextOutput("cindex_params")
          )
        ),
        fluidRow(
          box(title = "Détails de la partition", status = "info", 
              solidHeader = TRUE, width = 12,
              verbatimTextOutput("partition_info")
          )
        )
      )
    )
  )
)

# Serveur
server <- function(input, output, session) {
  
  # Réactif pour calculer les résultats
  results <- eventReactive(input$calculate, {
    set.seed(input$seed)
    
    # Créer la tâche de survie
    task <- TaskSurv$new(
      id = "veteran_lung", 
      backend = veteran, 
      time = "time", 
      event = "status"
    )
    
    # Partition train/test
    part <- partition(task, ratio = input$train_ratio)
    
    # Entraîner le modèle de Cox
    cox <- lrn("surv.coxph")
    cox$train(task, row_ids = part$train)
    
    # Prédictions sur le test set
    p <- cox$predict(task, row_ids = part$test)
    
    # Calculer C-index (Harrell uniquement, sans paramètres)
    cindex_measure <- msr("surv.cindex")
    cindex_value <- p$score(cindex_measure)
    
    # Calculer Kaplan-Meier pour la visualisation
    task_km <- TaskSurv$new(
      id = "veteran_km", 
      backend = veteran, 
      time = "time", 
      event = "status"
    )
    kaplan <- lrn("surv.kaplan")
    kaplan$train(task_km)
    pred_km <- kaplan$predict(task_km)
    surv_dist <- pred_km$distr
    times_unique <- sort(unique(veteran$time))
    surv_probs <- sapply(times_unique, function(t) {
      surv_dist$survival(t)[1]
    })
    df_km <- data.frame(
      time = times_unique,
      surv = surv_probs
    )
    
    list(
      cindex_value = cindex_value,
      df_km = df_km,
      part = part,
      task = task
    )
  })
  
  # Plot Kaplan-Meier
  output$km_plot <- renderPlot({
    req(results())
    df_km <- results()$df_km
    
    ggplot(df_km, aes(x = .data$time, y = .data$surv)) +
      geom_step(color = "darkred", linewidth = 1.2) +
      geom_ribbon(aes(ymin = 0, ymax = .data$surv), alpha = 0.3, fill = "lightcoral") +
      labs(
        title = "Courbe de Kaplan-Meier - Dataset Veterans Lung Cancer",
        x = "Temps (jours)",
        y = "Probabilité de survie"
      ) +
      theme_bw() +
      theme(plot.title = element_text(face = "bold", size = 14)) +
      scale_y_continuous(limits = c(0, 1))
  })
  
  # Informations sur le dataset
  output$dataset_info <- renderText({
    paste0(
      "Nombre total de patients: ", length(veteran$time), "\n",
      "Événements observés (décès): ", sum(veteran$status == 1), "\n",
      "Censures: ", sum(veteran$status == 0), "\n",
      "Temps médian: ", median(veteran$time), " jours\n",
      "Temps moyen: ", round(mean(veteran$time), 2), " jours"
    )
  })
  
  # Affichage C-index
  output$cindex_output <- renderText({
    req(results())
    paste0("C-index = ", round(results()$cindex_value, 4))
  })
  
  output$cindex_params <- renderText({
    req(results())
    "Harrell (par défaut, sans paramètres)"
  })
  
  # Informations sur la partition
  output$partition_info <- renderText({
    req(results())
    part <- results()$part
    task <- results()$task
    paste0(
      "Ratio train/test: ", input$train_ratio, "\n",
      "Taille train: ", length(part$train), " observations\n",
      "Taille test: ", length(part$test), " observations\n",
      "Seed utilisé: ", input$seed
    )
  })
}

# Lancer l'application
shinyApp(ui = ui, server = server)

