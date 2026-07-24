#import "@preview/cetz:0.4.2": canvas, draw

#set page(width: auto, height: auto, margin: 6pt, fill: rgb("#FFFFFF"))
#set text(font: "DejaVu Sans Mono")

#let ink = rgb("#0B1F33")
#let muted = rgb("#526579")
#let navy = rgb("#19324D")
#let lane-fill = rgb("#EEF3F7")
#let lane-rule = rgb("#B8C4D0")
#let cpu-fill = rgb("#A9D8F5")
#let cpu-border = rgb("#1E78B4")
#let gpu-fill = rgb("#FFAD33")
#let gpu-border = rgb("#C86500")
#let integration-fill = rgb("#B7EFC5")
#let integration-border = rgb("#168044")
#let boundary = rgb("#2457D6")

#canvas({
  import draw: *

  let lane(y0, y1, label) = {
    rect(
      (0.3, y0),
      (27.7, y1),
      fill: lane-fill,
      stroke: (paint: lane-rule, thickness: 0.7pt),
      radius: 0.18,
    )
    rect(
      (0.3, y0),
      (4.2, y1),
      fill: navy,
      stroke: none,
      radius: (west: 0.18, rest: 0),
    )
    content(
      (2.25, (y0 + y1) / 2),
      align(center, text(size: 11.5pt, weight: "bold", fill: white)[#label]),
    )
  }

  let phase(
    x0,
    x1,
    y0,
    y1,
    fill-color,
    border-color,
    label,
    text-size: 11pt,
  ) = {
    rect(
      (x0, y0),
      (x1, y1),
      fill: fill-color,
      stroke: (paint: border-color, thickness: 1.25pt),
      radius: 0.18,
    )
    content(
      ((x0 + x1) / 2, (y0 + y1) / 2),
      align(center, text(size: text-size, weight: "bold", fill: ink)[#label]),
    )
  }

  // Title and qualitative time axis.
  content(
    (14.0, 8.0),
    text(size: 17pt, weight: "bold", fill: ink)[General CPU/GPU event phases],
  )
  line(
    (4.35, 6.15),
    (27.75, 6.15),
    stroke: (paint: muted, thickness: 1.1pt),
    mark: (end: ">"),
  )
  content(
    (16.0, 6.5),
    text(size: 9.5pt, weight: "semibold", fill: muted)[relative time / phase order],
  )

  // Worker lanes.
  lane(3.55, 5.2, [CPU event-worker #linebreak() lane])
  lane(1.55, 3.2, [GPU worker #linebreak() lane])

  // Event boundaries. These delimit one event but do not define a clock scale.
  line(
    (4.35, 0.95),
    (4.35, 6.85),
    stroke: (paint: boundary, thickness: 2pt),
  )
  line(
    (27.25, 0.95),
    (27.25, 6.85),
    stroke: (paint: boundary, thickness: 2pt),
  )
  content(
    (4.35, 7.05),
    text(size: 10pt, weight: "bold", fill: boundary)[EVENT START],
  )
  content(
    (27.25, 7.05),
    text(size: 10pt, weight: "bold", fill: boundary)[EVENT COMPLETE],
  )

  // Phase bars. Both CPU phases intentionally share one vertical level.
  phase(
    4.55,
    12.95,
    3.82,
    4.93,
    cpu-fill,
    cpu-border,
    [Basket-forming CPU phase],
  )
  phase(
    12.95,
    20.15,
    1.82,
    2.93,
    gpu-fill,
    gpu-border,
    [GPU basket-processing phase],
  )
  phase(
    20.15,
    23.65,
    3.82,
    4.93,
    cpu-fill,
    cpu-border,
    [CPU continuation #linebreak() phase],
    text-size: 10pt,
  )
  phase(
    23.85,
    27.05,
    3.82,
    4.93,
    integration-fill,
    integration-border,
    [Event #linebreak() integration],
    text-size: 10pt,
  )

  // Submission and completion hand-offs in the blocking-style sequence.
  line(
    (12.95, 3.82),
    (12.95, 2.93),
    stroke: (paint: gpu-border, thickness: 1.1pt),
    mark: (end: ">"),
  )
  line(
    (20.15, 2.93),
    (20.15, 3.82),
    stroke: (paint: cpu-border, thickness: 1.1pt),
    mark: (end: ">"),
  )

  content(
    (16.0, 0.55),
    text(
      size: 9.5pt,
      style: "italic",
      fill: muted,
    )[Qualitative ordering only — horizontal widths do not encode durations.],
  )
})
