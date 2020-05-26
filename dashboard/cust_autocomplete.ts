import {AutocompleteInput, AutocompleteInputView} from "models/widgets/autocomplete_input"
import {Keys} from "core/dom"
import * as p from "core/properties"


export class ExtAutocompleteInputView extends AutocompleteInputView {
  model: ExtAutocompleteInput

  _keyup(event: KeyboardEvent): void {
    switch (event.keyCode) {
      case Keys.Enter: {
        this.change_input()
        break
      }
      case Keys.Esc: {
        this._hide_menu()
        break
      }
      case Keys.Up: {
        this._bump_hover(this._hover_index-1)
        break
      }
      case Keys.Down: {
        this._bump_hover(this._hover_index+1)
        break
      }
      default: {
        const value = this.input_el.value

        if (value.length < this.model.min_characters) {
          this._hide_menu()
          return
        }

        const completions: string[] = []
        const case_sensitive: boolean = this.model.case_sensitive
        let actest: (t: string, v: string) => boolean;
        if (case_sensitive) {
          actest = (t: string, v: string): boolean => { return t.startsWith(v); }
        } else {
          actest = (t: string, v: string): boolean => { return t.toLowerCase().startsWith(v.toLowerCase()); }
        }

        for (const text of this.model.completions) {
          if (actest(text, value)) {
            completions.push(text)
            }
        }

        this._update_completions(completions)
        if (completions.length == 0)
          this._hide_menu()
        else
          this._show_menu()
      }
    }
  }
}

export namespace ExtAutocompleteInput {
  export type Attrs = p.AttrsOf<Props>

  export type Props = AutocompleteInput.Props & {
    case_sensitive: p.Property<boolean>
  }
}

export interface ExtAutocompleteInput extends ExtAutocompleteInput.Attrs {}

export class ExtAutocompleteInput extends AutocompleteInput {
  properties: ExtAutocompleteInput.Props

  constructor(attrs?: Partial<ExtAutocompleteInput.Attrs>) {
    super(attrs)
  }

  static init_ExtAutocompleteInput(): void {
    this.prototype.default_view = ExtAutocompleteInputView

    this.define<ExtAutocompleteInput.Props>({
      case_sensitive: [ p.Boolean,  false ],
    })
  }
}