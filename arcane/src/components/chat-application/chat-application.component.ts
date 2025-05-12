import { CommonModule } from '@angular/common';
import { Component, inject, OnDestroy, OnInit } from '@angular/core';
import { FormControl, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';

import { ChatService } from '../../services/chat.service';
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
  chatService = inject(ChatService);
  wsService = inject(WebsocketService);
  answer: string = '';

  question = '';
  isLoading = false;
  error = false;
  character = 'dumbledore';
  includeSources = false;

  formGroup = new FormGroup({
    prompt: new FormControl('', Validators.required),
    tone: new FormControl('', Validators.required),
    includeSources: new FormControl(false),
  });

  buffer = '';

  constructor() {
    this.displayRandomBooks();
  }

  ngOnInit() {
    this.wsService.connect();
  
    this.wsService.messages$.subscribe((data: string) => {
      if (data === '__END__') {
        this.answer = this.buffer;
        this.buffer = '';
      } else {
        this.buffer += data;
      }
    });
  }
  
  ngOnDestroy() {
    this.wsService.close();
  }

  generateResponse(voice: boolean): void {
    this.isLoading = true;
    const formValue = this.formGroup.value ?? {};

    this.wsService.sendMessage(JSON.stringify(formValue), voice);

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
}
