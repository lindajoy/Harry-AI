import { CommonModule } from '@angular/common';
import { Component, inject, OnDestroy, OnInit } from '@angular/core';
import { FormControl, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';

import { PromptService } from '../../services/chat.service';
import { WebsocketService } from '../../services/web.socket.service';

@Component({
  selector: 'app-chat-application',
  standalone: true,
  imports: [CommonModule, FormsModule, ReactiveFormsModule],
  templateUrl: './chat-application.component.html',
  styleUrl: './chat-application.component.css'
})
export class ChatApplicationComponent implements OnInit, OnDestroy {
  books: string[] = ['Half Blood Prince', 'Deathly Hallows', 'Soceerers Stone', 'Goblet of fire', 'Order of the phoneix'];
  displayedBooks: string[] = [];
  chatService = inject(PromptService);
  wsService = inject(WebsocketService);
  answer: string = '';

  promptStyles$ = this.chatService.generatePrompts();

  question = '';
  isLoading = false;
  error = false;
  character = 'dumbledore';
  includeSources = false;

  formGroup = new FormGroup({
    prompt: new FormControl('', Validators.required),
    tone: new FormControl('Albus Dumbledore', Validators.required),
    includeSources: new FormControl(false),
  });

  buffer = '';
  private previousPrompt = ""


  constructor() {
    this.displayRandomBooks();
  }

  ngOnInit() {
    this.wsService.connect(this.formGroup?.value?.tone ?? 'Albus Dumbledore');

    this.wsService.messages$.subscribe((data: string) => {
      if (data === '__END__') {
        this.answer = this.buffer;
        this.buffer = '';
      } else {
        this.buffer += data;
      }
    });

    this.formGroup.get("prompt")?.valueChanges.subscribe((value) => {
      const currentPrompt = value || ""

      if (this.answer && this.previousPrompt && currentPrompt !== this.previousPrompt) {
        this.answer = "" 
      }
      this.previousPrompt = currentPrompt
    })
  }
 
  generateResponse(): void {
    this.isLoading = true;
    const formValue = this.formGroup.value ?? {};

    this.wsService.sendMessage(JSON.stringify(formValue.prompt), formValue.tone as string);

    const subscription = this.wsService.messages$
                               .subscribe(response => {
                                  this.isLoading = false;
                                  subscription.unsubscribe();
                            });
  }

  resetError() {
    this.error = false;
  }

  handleClear(): void {
    this.question = '';
    this.answer = '';
  }

  displayRandomBooks() {
    const shuffled = [...this.books].sort(() => 0.5 - Math.random());
    this.displayedBooks = shuffled.slice(0, 3);
  }

  ngOnDestroy() {
    this.wsService.close();
  }
}
