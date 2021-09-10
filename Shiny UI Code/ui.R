library(shiny)
library(DT)
library(jpeg)
library(imager)
library(mxnet)
library(magrittr)
library(OpenImageR)
library(pROC)
library(rlist)

shinyUI(
  navbarPage("FaceNet",
             #Start Page1       
             tabPanel("Manual Marking System",
              fluidPage(
                fluidRow(
                  column(width = 4,
                         radioButtons("mark_pic", "Select the source of picture:", choices = c( 
                                                                                              "Example" = "Example",
                                                                                              "Upload" = "Upload")),
                         fileInput("files", label = h4("Upload your jpeg image:"), multiple = FALSE, accept = "image/jpeg"),
                         br(),
                         textInput("name", "Input name here"),
                         br(),
                         actionButton("Save", "Save to database"),
                         br(),
                         br(),
                         downloadButton("download", label = "Download file", class = NULL)
                  ),
                  column(width = 5,
                         #Resize
                         plotOutput("plot", height = 416, width = 416,
                                    dblclick = "plot_dblclick",
                                    brush = brushOpts(id = "plot_brush", resetOnNew = TRUE)),
                         br(),
                         actionButton("delete", strong("Delete selected box!"), icon("list-alt")),
                         actionButton("Show", "Show cropped face"),
                         br(),
                         br(),
                         plotOutput("box_plot", height = 72, width = 72),
                         br(),
                         DT::dataTableOutput('table')
                  )
                )
              )
             ),
             #Start Page2
             tabPanel("Face Detection",
                      fluidPage(
                        fluidRow(
                          column(width = 4,
                                 radioButtons("Picture", "Select the source of picture:", choices = c("Upload" = "Upload",
                                                                                                      "Author1" = 'yj',
                                                                                                      "Author2" = 'yc3',
                                                                                                      "Instructor" = 'linchin',
                                                                                                      "Man1" = "man1",
                                                                                                      "Woman1" = "woman1"
                                                                                                      )),
                                 fileInput("detection_files", label = h4("Upload your Human jpeg image:"), multiple = FALSE, accept = "image/jpeg")
                          ),
                          column(width = 5,
                                 br(),
                                 actionButton("Show2", "Show Result"),
                                 plotOutput("prediction_image", height = 384, width = 384),
                                 br(),
                                 textOutput("class_person"),
                                 br(),
                                 br(),
                                 DT::dataTableOutput('Summary'),
                                 br()
                          )
                        )
                      )
             ),
             #Start Page3
             tabPanel("Face Database",
                      fluidPage(
                        fluidRow(
                          column(width = 11,
                                 br(),
                                 DT::dataTableOutput('face_dataframe')
                                 # actionButton("delete_base", strong("Delete selected box!"), icon("list-alt"))
                          )
                        )
                      )
             ),
             #Start Page4
             tabPanel("Author",
                      fluidPage(
                        fluidRow(
                          column(width = 3,
                                 br(),
                                 plotOutput("linchin_image", height = 384, width = 384),
                                 br(),
                                 br(),
                                 plotOutput("yc_image", height = 384, width = 384),
                                 br(),
                                 br(),
                                 plotOutput("yj_image", height = 384, width = 384),
                          ),
                          column(width = 5,
                                 br(),
                                 verbatimTextOutput("linchin_text", placeholder = FALSE),
                                 br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),
                                 verbatimTextOutput("yc_text", placeholder = FALSE),
                                 br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),
                                 verbatimTextOutput("yj_text", placeholder = FALSE)

                          )
                        )
                      )
             )
             
  )
)