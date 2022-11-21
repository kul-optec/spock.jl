using Documenter, spock


makedocs(
  sitename="Spock.jl",
  pages = [
    "Home" => "index.md",
    "Examples" => "examples.md",
  ]
)

deploydocs(
  repo = "github.com/kul-optec/spock.jl.git"
)